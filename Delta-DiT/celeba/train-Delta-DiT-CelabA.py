import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import os
import time
import logging
import math
import copy
import threading
import torchvision
from torchvision.utils import make_grid
from PIL import Image
from datetime import datetime




class Config:
    
    exp_name = "DeltaDiT_CelebA64_XPred"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"{exp_name}_{run_id}")

    
    data_path = "data/celeba_64x64.pt"
    image_size = 64
    patch_size = 4  

    
    hidden_size = 128
    depth = 34
    num_heads = 4
    mlp_ratio = 4.0

    
    batch_size = 16  
    lr = 2e-4  
    epochs = 100
    grad_clip = 1.0
    ema_decay = 0.999

    
    sample_every_steps = 100  
    log_every_steps = 10
    save_every_epochs = 5
    num_sample_images = 4  
    ode_steps = 20  

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0



os.makedirs(os.path.join(Config.output_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(Config.output_dir, "samples"), exist_ok=True)
os.makedirs(os.path.join(Config.output_dir, "checkpoints"), exist_ok=True)





def setup_logger(log_dir):
    logger = logging.getLogger("DeltaDiT")
    logger.setLevel(logging.INFO)

    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    
    fh = logging.FileHandler(os.path.join(log_dir, "training_log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logger(os.path.join(Config.output_dir, "logs"))









class ProxyDeltaLinear(nn.Module):
    """ Delta 算子: 基于 L1/L2 距离的线性投影 """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.gain = nn.Parameter(torch.ones(1, out_dim, 1) * 2.0)

    def forward(self, x):
        
        x_act = torch.tanh(x)
        w_act = torch.tanh(self.weight)

        B, T, Din = x_act.shape
        x_flat = x_act.reshape(-1, Din)

        
        
        l1_sum = torch.cdist(x_flat, w_act, p=1)
        l2_euc = torch.cdist(x_flat, w_act, p=2)
        l2_sq_sum = torch.square(l2_euc)

        l1_mean = l1_sum / Din
        l2_sq_mean = l2_sq_sum / Din

        
        dist_proxy = (l1_mean - l2_sq_mean).detach() + l2_sq_mean

        
        out = 1.0 - dist_proxy
        out = out.view(B, T, self.out_dim)
        out = (out - 0.5) * self.gain.view(1, 1, -1)
        return out


class ProxyStudentTAttention(nn.Module):
    """ Delta Attention: 基于 Student-T 核的距离注意力 """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        
        self.w_q = ProxyDeltaLinear(d_model, d_model)
        self.w_k = ProxyDeltaLinear(d_model, d_model)
        self.w_v = ProxyDeltaLinear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)  

        
        self.gamma = nn.Parameter(torch.ones(1, n_heads, 1, 1) * 1.0)
        self.rho = nn.Parameter(torch.ones(1, n_heads, 1, 1) * 2.0)

    def forward(self, x):
        B, T, C = x.shape

        q = self.w_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  
        k = self.w_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        
        q_flat = q.reshape(B * self.n_heads, T, self.d_head)
        k_flat = k.reshape(B * self.n_heads, T, self.d_head)

        l1_sum = torch.cdist(q_flat, k_flat, p=1)
        l2_euc = torch.cdist(q_flat, k_flat, p=2)

        l1_mean = l1_sum / self.d_head
        l2_sq = torch.square(l2_euc) / self.d_head

        dist_proxy = (l1_mean - l2_sq).detach() + l2_sq
        dist_proxy = dist_proxy.view(B, self.n_heads, T, T)

        
        gamma_pos = F.softplus(self.gamma) + 1e-4
        rho_pos = F.softplus(self.rho) + 1e-4

        
        base_kernel = 1.0 / (1.0 + gamma_pos * dist_proxy)
        attn_weights = torch.pow(base_kernel, rho_pos)

        
        attn_probs = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)

        out = attn_probs @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.w_o(out)





class DeltaDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = ProxyStudentTAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            ProxyDeltaLinear(hidden_size, mlp_hidden_dim),
            approx_gelu(),
            ProxyDeltaLinear(mlp_hidden_dim, hidden_size),
        )

        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        x_norm1 = self.modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm1)

        x_norm2 = self.modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm2)
        return x

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DeltaDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = 3
        self.out_channels = 3 * (config.patch_size ** 2)
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size

        
        self.x_embedder = nn.Conv2d(3, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)

        
        self.t_embedder = nn.Sequential(
            nn.Linear(256, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))

        
        self.blocks = nn.ModuleList([
            DeltaDiTBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
            for _ in range(config.depth)
        ])

        
        self.final_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(config.hidden_size, self.out_channels, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        
        nn.init.normal_(self.pos_embed, std=0.02)
        
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x):
        c = 3
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t):
        
        x = self.x_embedder(x)  
        x = x.flatten(2).transpose(1, 2)  
        x = x + self.pos_embed

        
        t_emb = self.get_timestep_embedding(t, 256).to(x.device)
        c = self.t_embedder(t_emb)

        
        for block in self.blocks:
            x = block(x, c)

        
        x = self.final_norm(x)
        x = self.final_linear(x)
        x = self.unpatchify(x)
        return x

    def get_timestep_embedding(self, timesteps, embedding_dim):
        assert len(timesteps.shape) == 1
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb



class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model)
        self.decay = decay
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)


def get_dataloader():
    if not os.path.exists(Config.data_path):
        raise FileNotFoundError("Run the download script first!")
    print(f"Loading data from {Config.data_path}...")
    data = torch.load(Config.data_path, map_location='cpu')
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=Config.batch_size, shuffle=True,
                      num_workers=Config.num_workers, pin_memory=True, drop_last=True)



class AsyncSampler(threading.Thread):
    def __init__(self, model_state_dict, step):
        super().__init__()
        self.model_state_dict = model_state_dict
        self.step = step
        self.config = Config

    def run(self):
        
        
        
        try:
            device = self.config.device

            
            model = DeltaDiT(self.config).to(device)
            model.load_state_dict(self.model_state_dict)
            model.eval()

            self.sample_sequence(model, self.step, device)

            
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Async sampling failed: {e}")

    @torch.no_grad()
    def sample_sequence(self, model, step, device):
        n = self.config.num_sample_images
        z = torch.randn(n, 3, 64, 64, device=device)
        snapshot_timesteps = np.linspace(0, 1, 8)
        snapshots = []


        dt = 1.0 / self.config.ode_steps
        current_t = 0.0

        for i in range(self.config.ode_steps):
            
            relative_t = i / self.config.ode_steps
            
            if any(abs(relative_t - s) < (0.5 * dt) for s in snapshot_timesteps):
                snapshots.append(z.cpu())
            
            t_tensor = torch.ones(n, device=device) * current_t
            x_pred = model(z, t_tensor)

            denom = 1 - current_t
            if denom < 1e-5:
                denom = 1e-5

            eps_pred = (z - current_t * x_pred) / denom
            v_pred = x_pred - eps_pred

            z = z + v_pred * dt
            current_t += dt

        snapshots.append(z.cpu())

        

        rows = []
        for i in range(n):
            row_imgs = []
            for snap in snapshots:
                
                img = snap[i].clamp(-1, 1) * 0.5 + 0.5
                row_imgs.append(img)
            rows.append(torch.cat(row_imgs, dim=2))  

        final_grid = torch.cat(rows, dim=1)  

        
        save_path = os.path.join(self.config.output_dir, "samples", f"step_{step:06d}.png")
        torchvision.utils.save_image(final_grid, save_path)
        print(f"[Async] Sample saved to {save_path}")





def train():
    print(f"🚀 Starting Experiment: {Config.exp_name}")
    print(f"📂 Output Dir: {Config.output_dir}")

    
    dataloader = get_dataloader()
    print(f"📊 Dataset Size: {len(dataloader.dataset)}")

    
    model = DeltaDiT(Config).to(Config.device)
    ema = EMA(model, Config.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=0.0)
    scaler = GradScaler(device=Config.device.type)

    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {total_params / 1e6:.2f} M")

    
    global_step = 0
    best_loss = float('inf')

    model.train()

    for epoch in range(Config.epochs):
        logger.info(f"Epoch {epoch + 1}/{Config.epochs} Started")

        start_time = time.time()
        loss_accum = 0.0

        for batch_idx, (x,) in enumerate(dataloader):
            x = x.to(Config.device)
            B = x.shape[0]
            s = torch.randn(B, device=Config.device) * 0.8 - 0.8
            t = torch.sigmoid(s)
            eps = torch.randn_like(x)

            t_b = t.view(B, 1, 1, 1)
            z_t = t_b * x + (1 - t_b) * eps

            with autocast(device_type=Config.device.type):
                x_pred = model(z_t, t)
                loss = F.mse_loss(x_pred, x)

            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            loss_accum += loss.item()
            global_step += 1

            
            if global_step % Config.log_every_steps == 0:
                
                with torch.no_grad():
                    
                    gamma_mean = model.blocks[0].attn.gamma.mean().item()
                    rho_mean = model.blocks[0].attn.rho.mean().item()

                logger.info(f"Step {global_step} | Loss: {loss.item():.6f} | "
                            f"Grad: {grad_norm:.4f} | "
                            f"Gamma: {gamma_mean:.3f} | Rho: {rho_mean:.3f}")

            
            if global_step % Config.sample_every_steps == 0:
                
                ema_state = {k: v.cpu() for k, v in ema.model.state_dict().items()}
                sampler_thread = AsyncSampler(ema_state, global_step)
                sampler_thread.start()
                

        
        avg_loss = loss_accum / len(dataloader)
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch + 1} Done. Avg Loss: {avg_loss:.6f}. Time: {epoch_time:.1f}s")

        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ema.model.state_dict(), os.path.join(Config.output_dir, "checkpoints", "best_model.pt"))
            logger.info(f"🌟 Best Model Saved! Loss: {best_loss:.6f}")

        
        if (epoch + 1) % Config.save_every_epochs == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema': ema.model.state_dict(),
            }, os.path.join(Config.output_dir, "checkpoints", f"epoch_{epoch + 1}.pt"))


if __name__ == "__main__":
    
    torch.backends.cudnn.benchmark = True
    train()