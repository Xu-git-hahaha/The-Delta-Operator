import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
VOCAB_SIZE = tokenizer.vocab_size
print("Vocab size:", VOCAB_SIZE)




BLOCK_SIZE = 64
BATCH_SIZE = 4
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 15

Delta_SCALE = 1.0
DIR_SCALE = 0.5
ATTN_SCALE = 1.0
DROPOUT = 0.05

LR = 1.0e-3
MAX_ITERS = 31000
EVAL_INTERVAL = 100
MAX_STORIES = 300000
DATA_DIR = "data/wikitext-103"
MODEL_PATH = "Delta_v68_scheduler_dropout0.05_100M_wiki.pt"





def causal_mask(T, device):
    return torch.tril(torch.ones(T, T, device=device))


def causal_shift(x, shift):
    B, T, D = x.shape
    pad = torch.zeros(B, shift, D, device=x.device)
    return torch.cat([pad, x[:, :-shift, :]], dim=1)


def load_data_tensor(split):
    
    path = os.path.join(DATA_DIR, f"{split}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please run 'prepare_wikitext.py' first.")

    print(f"📂 Loading {split} data from {path}...")
    
    return torch.load(path, map_location=DEVICE)


print("Initializing Data Loaders (WikiText-103)...")

try:
    
    train_data = load_data_tensor("train")
    val_data = load_data_tensor("validation")

    print(f"✅ Data Loaded Successfully!")
    print(f"   Train tokens: {len(train_data) / 1e6:.2f} M")
    print(f"   Val tokens:   {len(val_data) / 1e6:.2f} M")

except FileNotFoundError as e:
    print(f"⚠️ Critical Error: {e}")
    print("➡️ Fallback: Generating random noise for debugging (Model will not learn!)")
    train_data = torch.randint(0, VOCAB_SIZE, (10000,), device=DEVICE)
    val_data = torch.randint(0, VOCAB_SIZE, (1000,), device=DEVICE)


def get_batch(data):
    
    
    
    if data is train_data:
        print(f"\n✅ [AUDIT] get_batch is using: TRAIN_DATA (Size: {len(data)})")
    elif data is val_data:
        print(f"\n🧐 [AUDIT] get_batch is using: VAL_DATA (Size: {len(data)})")
    else:
        
        print(f"\n❓ [AUDIT] get_batch is using: UNKNOWN DATA (Size: {len(data)})")
    

    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])

    
    return x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)






class ProxyDeltaLinearV68(nn.Module):
    """
    V68 Ultimate Optimization:
    使用 torch.cdist 替代手动广播。
    cdist 底层是高度优化的 C++ 内核，且 p=2 时能利用 Tensor Cores。
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.gain = nn.Parameter(torch.ones(1, out_dim, 1) * 2.0)

    def forward(self, x):
        
        

        
        x_act = torch.tanh(x)
        w_act = torch.tanh(self.weight)

        B, T, Din = x_act.shape
        Dout = self.out_dim

        
        
        x_flat = x_act.reshape(-1, Din)

        
        
        

        
        l1_sum = torch.cdist(x_flat, w_act, p=1)

        
        
        l2_euc = torch.cdist(x_flat, w_act, p=2)
        l2_sq_sum = torch.square(l2_euc)

        
        
        

        l1_mean = l1_sum / Din
        l2_sq_mean = l2_sq_sum / Din

        
        dist_proxy_mean = (l1_mean - l2_sq_mean).detach() + l2_sq_mean

        
        out = 1.0 - dist_proxy_mean

        
        out = out.view(B, T, Dout)

        
        out = (out - 0.5) * self.gain.view(1, 1, -1)
        return out


class ProxyStudentTAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = ProxyDeltaLinearV68(d_model, d_model)
        self.w_k = ProxyDeltaLinearV68(d_model, d_model)
        self.w_v = ProxyDeltaLinearV68(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.gamma = nn.Parameter(torch.ones(1, n_heads, 1, 1) * 1.0)
        self.rho = nn.Parameter(torch.ones(1, n_heads, 1, 1) * 2.0)
        self.scale = math.sqrt(self.d_head)
        self.register_buffer('last_entropy', torch.tensor(0.0))

    def forward(self, x, mask=None):
        B, T, C = x.shape

        
        q = torch.tanh(self.w_q(x)).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  
        k = torch.tanh(self.w_k(x)).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  
        v = self.w_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        
        
        
        

        q_flat = q.reshape(B * self.n_heads, T, self.d_head)
        k_flat = k.reshape(B * self.n_heads, T, self.d_head)

        
        
        l1_sum = torch.cdist(q_flat, k_flat, p=1)
        l2_euc = torch.cdist(q_flat, k_flat, p=2)
        l2_sq_sum = torch.square(l2_euc)

        
        l1_mean = l1_sum / self.d_head
        l2_sq_mean = l2_sq_sum / self.d_head

        dist_proxy = (l1_mean - l2_sq_mean).detach() + l2_sq_mean

        
        dist_proxy = dist_proxy.view(B, self.n_heads, T, T)

        dist_scaled = dist_proxy * self.scale
        dist_final_sq = torch.square(dist_scaled)

        gamma_pos = F.softplus(self.gamma)
        rho_pos = F.softplus(self.rho)

        base_kernel = 1.0 / (1.0 + gamma_pos * dist_final_sq + 1e-6)
        attn_weights = torch.pow(base_kernel, rho_pos)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, 0.0)

        attn_sum = attn_weights.sum(dim=-1, keepdim=True) + 1e-6
        attn_probs = attn_weights / attn_sum

        with torch.no_grad():
            self.last_entropy = -(attn_probs * torch.log(attn_probs + 1e-9)).sum(-1).mean()

        out = attn_probs @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.w_o(out)


class DeltaSwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.hidden_dim = d_model * 4
        self.Delta_gate = ProxyDeltaLinearV68(d_model, self.hidden_dim)
        self.Delta_val = ProxyDeltaLinearV68(d_model, self.hidden_dim)
        self.Delta_out = ProxyDeltaLinearV68(self.hidden_dim, d_model)

    def forward(self, x):
        gate = F.silu(self.Delta_gate(x))
        val = self.Delta_val(x)
        x = gate * val
        return self.Delta_out(x)


class Direction(nn.Module):
    def __init__(self, d_model, shift):
        super().__init__()
        self.shift = shift
        self.w = nn.Parameter(torch.randn(d_model) * 0.1)

    def forward(self, x):
        xp = causal_shift(x, self.shift)
        w = torch.sigmoid(self.w).view(1, 1, -1)
        return torch.abs(xp - w)


class DeltaBlock(nn.Module):
    def __init__(self, d_model, heads, shift):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ProxyStudentTAttention(d_model, heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = DeltaSwiGLU(d_model)
        self.dir = Direction(d_model, shift)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return checkpoint(self._forward_impl, x, use_reentrant=True)

    def _forward_impl(self, x):
        mask = causal_mask(x.size(1), x.device)
        a = self.attn(self.ln1(x), mask=mask)
        x = x + a * ATTN_SCALE
        f = self.ffn(self.ln2(x))
        d = self.dir(self.ln2(x))
        x = x + (f * Delta_SCALE + d * DIR_SCALE)
        return self.dropout(x)


class DeltaGPT_V68(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(BLOCK_SIZE, D_MODEL)
        shifts = [1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2]
        self.blocks = nn.Sequential(*[DeltaBlock(D_MODEL, N_HEADS, shifts[i]) for i in range(N_LAYERS)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.Delta_bridge = ProxyDeltaLinearV68(D_MODEL, 256)
        self.bridge_ln = nn.LayerNorm(256)
        self.head = nn.Linear(256, VOCAB_SIZE)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token(idx) + self.pos(pos)

        if not x.requires_grad:
            x.requires_grad_(True)

        x = self.blocks(x)
        x = self.ln(x)
        x = self.Delta_bridge(x)
        x = self.bridge_ln(x)
        return self.head(x)






def model_structure_analysis(model):
    total_params = 0
    embedding_params = 0
    head_params = 0
    backbone_params = 0
    Delta_params_in_backbone = 0

    Delta_module_ids = set()
    for name, module in model.named_modules():
        if isinstance(module, (ProxyDeltaLinearV68, Direction)):
            Delta_module_ids.add(id(module))

    for name, param in model.named_parameters():
        num = param.numel()
        total_params += num

        is_embedding = "token" in name or "pos" in name
        is_head = "head" in name

        if is_embedding:
            embedding_params += num
        elif is_head:
            head_params += num
        else:
            backbone_params += num

            is_Delta = False
            for m in model.modules():
                if id(m) in Delta_module_ids:
                    for p in m.parameters(recurse=False):
                        if id(p) == id(param):
                            is_Delta = True
                            break
                if is_Delta: break

            if is_Delta:
                Delta_params_in_backbone += num

    print("\n" + "=" * 60)
    print("📝 PAPER ANALYSIS REPORT: MODEL ARCHITECTURE (DeltaGPT)")
    print("=" * 60)
    print(f"Total Parameters         : {total_params / 1e6:.2f} M")
    print("-" * 30)
    print(f"Embedding Params         : {embedding_params / 1e6:.2f} M")
    print(f"Head Params (Final Proj) : {head_params / 1e6:.2f} M")
    print("-" * 30)
    print(f"Backbone Params (Computation Core) : {backbone_params / 1e6:.2f} M")
    print(f"  ├─ Delta Params (L1/XOR-like)    : {Delta_params_in_backbone / 1e6:.2f} M")
    print(f"  └─ Standard Params (Norm/Linear) : {(backbone_params - Delta_params_in_backbone) / 1e6:.2f} M")
    print("-" * 30)
    if backbone_params > 0:
        Delta_ratio = (Delta_params_in_backbone / backbone_params) * 100
        print(f"👉 Delta Parameter Ratio in Backbone : {Delta_ratio:.2f}%")
    print("=" * 60 + "\n")


@torch.no_grad()
def evaluate_metrics(model, data, iters=20):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    total_acc = 0

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for _ in range(iters):
        xb, yb = get_batch(data)
        with autocast(device_type='cuda', dtype=amp_dtype):
            out = model(xb)
            loss = loss_fn(out.view(-1, VOCAB_SIZE), yb.view(-1))

        total_loss += loss.item()
        acc = (out.argmax(-1) == yb).float().mean().item()
        total_acc += acc

    model.train()
    return total_loss / iters, total_acc / iters


def train():
    torch.set_float32_matmul_precision('high')

    model = DeltaGPT_V68().to(DEVICE)
    model_structure_analysis(model)

    print("🚀 Optimization: cdist Kernel (Memory Efficient & TensorCore Ready)")
    print("⚠️ Skipping torch.compile (Windows Safe)")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_ITERS, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    scaler = GradScaler(device='cuda')
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"🚀 Using AMP dtype: {amp_dtype}")

    best_acc = 0

    print("Starting Training...")

    for step in range(MAX_ITERS):
        print(f"{step}.", end='')
        xb, yb = get_batch(train_data)

        
        with autocast(device_type='cuda', dtype=amp_dtype):
            out = model(xb)
            loss = loss_fn(out.view(-1, VOCAB_SIZE), yb.view(-1))

        
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        scheduler.step()

        if step % 50 == 0:
            print("\n")
            acc = (out.detach().float().argmax(-1) == yb).float().mean().item()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {step:05d} | Loss {loss.item():.4f} | TrainAcc {acc * 100:.2f}% | LR {current_lr:.5f}")

            if step % 50 == 0:
                print(f"  [Stats] Detailed Layer Report:")
                
                blocks_to_log = model.blocks if hasattr(model, 'blocks') else model._orig_mod.blocks

                for i, block in enumerate(blocks_to_log):
                    gamma = block.attn.gamma.mean().item()
                    rho = block.attn.rho.mean().item()
                    entropy_val = block.attn.last_entropy.item()
                    print(f"  [Stats] Gamma: {gamma:.3f} | [Stats] Rho: {rho:.3f}")
                    print(f"  [Stats] Block {i + 1} Entropy: {entropy_val:.3f}")

        if step % EVAL_INTERVAL == 0 and step > 0:
            val_loss, val_acc = evaluate_metrics(model, val_data)
            print(f"\n>>> [EVAL] Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                try:
                    save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                    torch.save(save_model.state_dict(), MODEL_PATH)
                    print(f"✅ Model Saved (Best Acc: {best_acc * 100:.2f}%)")
                except Exception as e:
                    print(f"Save failed: {e}")
            print("-" * 40)


if __name__ == "__main__":
    train()