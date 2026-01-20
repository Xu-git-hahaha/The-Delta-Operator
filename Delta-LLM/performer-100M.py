import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os




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

DROPOUT = 0.05


LR = 1.0e-3
MAX_ITERS = 31000
EVAL_INTERVAL = 100
MAX_STORIES = 300000
DATA_DIR = "data/wikitext-103"

MODEL_PATH = "baseline_performer_scheduler_dropout0.05_100M.pt"





def causal_mask(T, device):
    return torch.tril(torch.ones(T, T, device=device))


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
    
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])

    
    return x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)




def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = q.to(device), r.to(device)
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []
    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])
    final_matrix = torch.cat(block_list)
    multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    return torch.diag(multiplier) @ final_matrix






class PerformerAttention(nn.Module):
    """
    Performer Attention (FAVOR+) - Softmax kernel approximation.
    使用随机特征近似 softmax(QK^T / sqrt(d))，并保持线性复杂度 O(T).
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        
        self.m = self.d_head

        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        
        self.register_buffer(
            "proj_matrix",
            gaussian_orthogonal_random_matrix(self.m, self.d_head)
        )

        self.last_entropy = 0.0

    def _softmax_feature_map(self, x):
        """
        FAVOR+ softmax kernel 的随机特征映射：
        phi(x) ≈ exp( -||x||^2 / 2 ) * exp( x W )
        再做一些数值稳定处理。
        x: [B, H, T, D]
        返回: [B, H, T, M]，且全部为正数
        """
        B, H, T, D = x.shape
        proj = self.proj_matrix.type_as(x).t()  

        
        data_normalizer = 1.0 / math.sqrt(math.sqrt(D))
        x_scaled = x * data_normalizer

        
        x_proj = torch.einsum("bhtd,dm->bhtm", x_scaled, proj)

        
        diag = (x_scaled ** 2).sum(dim=-1, keepdim=True) / 2.0  
        x_proj = x_proj - diag

        
        x_proj = x_proj - x_proj.max(dim=-1, keepdim=True).values

        
        return torch.exp(x_proj) + 1e-6

    def forward(self, x, mask=None):
        
        B, T, C = x.shape
        H = self.n_heads
        D = self.d_head
        M = self.m

        
        q = self.w_q(x).view(B, T, H, D).transpose(1, 2)  
        k = self.w_k(x).view(B, T, H, D).transpose(1, 2)  
        v = self.w_v(x).view(B, T, H, D).transpose(1, 2)  

        
        q_prime = self._softmax_feature_map(q)  
        k_prime = self._softmax_feature_map(k)  

        
        
        

        
        
        
        kv = torch.einsum("bhtm,bhtd->bhtmd", k_prime, v)

        kv_cumsum = torch.cumsum(kv, dim=2)        
        k_cumsum = torch.cumsum(k_prime, dim=2)    

        
        num = torch.einsum("bhtm,bhtmd->bhtd", q_prime, kv_cumsum)

        
        den = torch.einsum("bhtm,bhtm->bht", q_prime, k_cumsum).unsqueeze(-1)
        den = den + 1e-6  

        out = num / den  

        
        with torch.no_grad():
            q_s = q_prime[0, 0]  
            k_s = k_prime[0, 0]  
            attn_approx = q_s @ k_s.t()  

            
            mask_s = torch.tril(torch.ones(T, T, device=x.device))
            attn_approx = attn_approx.masked_fill(mask_s == 0, 0.0)

            row_sums = attn_approx.sum(dim=-1, keepdim=True) + 1e-6
            attn_probs = attn_approx / row_sums

            self.last_entropy = float(
                (-(attn_probs * torch.log(attn_probs + 1e-9)).sum(-1).mean()).cpu()
            )

        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.w_o(out)



class StandardSwiGLU(nn.Module):
    """
    Standard SwiGLU FFN (Unchanged from Baseline)
    """

    def __init__(self, d_model):
        super().__init__()
        self.hidden_dim = d_model * 4
        self.gate = nn.Linear(d_model, self.hidden_dim)
        self.val = nn.Linear(d_model, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, d_model)

    def forward(self, x):
        gate = F.silu(self.gate(x))
        val = self.val(x)
        x = gate * val
        return self.out(x)


class PerformerBlock(nn.Module):
    """
    Replaces StandardBlock, using PerformerAttention
    """

    def __init__(self, d_model, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = PerformerAttention(d_model, heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = StandardSwiGLU(d_model)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        
        a = self.attn(self.ln1(x), mask=causal_mask(x.size(1), x.device))
        x = x + a

        f = self.ffn(self.ln2(x))
        x = x + f

        return self.dropout(x)


class BaselineGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(BLOCK_SIZE, D_MODEL)

        
        
        self.blocks = nn.Sequential(*[PerformerBlock(D_MODEL, N_HEADS) for _ in range(N_LAYERS)])

        self.ln = nn.LayerNorm(D_MODEL)

        
        
        self.bridge_proj = nn.Linear(D_MODEL, 256)
        self.bridge_ln = nn.LayerNorm(256)

        self.head = nn.Linear(256, VOCAB_SIZE)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token(idx) + self.pos(pos)
        x = self.blocks(x)
        x = self.ln(x)

        
        x = self.bridge_proj(x)
        x = self.bridge_ln(x)

        return self.head(x)






def model_structure_analysis(model):
    """
    Calculates parameter breakdown for Performer for fair comparison table.
    """
    total_params = 0
    embedding_params = 0
    head_params = 0
    backbone_params = 0

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

    print("\n" + "=" * 60)
    print("📝 PAPER ANALYSIS REPORT: PERFORMER MODEL ARCHITECTURE")
    print("=" * 60)
    print(f"Total Parameters         : {total_params / 1e6:.2f} M")
    print("-" * 30)
    print(f"Embedding Params         : {embedding_params / 1e6:.2f} M")
    print(f"Head Params (Final Proj) : {head_params / 1e6:.2f} M")
    print("-" * 30)
    print(f"Backbone Params (Linear Attn)      : {backbone_params / 1e6:.2f} M")
    print("=" * 60 + "\n")


@torch.no_grad()
def evaluate_metrics(model, data, iters=20):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    total_acc = 0

    for _ in range(iters):
        xb, yb = get_batch(data)
        out = model(xb)
        loss = loss_fn(out.view(-1, VOCAB_SIZE), yb.view(-1))
        total_loss += loss.item()
        acc = (out.argmax(-1) == yb).float().mean().item()
        total_acc += acc

    model.train()
    return total_loss / iters, total_acc / iters


def train():
    model = BaselineGPT().to(DEVICE)

    
    model_structure_analysis(model)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_ITERS, eta_min=1e-5)

    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0

    print("Starting Performer Baseline Training...")

    for step in range(MAX_ITERS):
        xb, yb = get_batch(train_data)
        out = model(xb)
        loss = loss_fn(out.view(-1, VOCAB_SIZE), yb.view(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        scheduler.step()

        if step % 50 == 0:
            acc = (out.argmax(-1) == yb).float().mean().item()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {step:05d} | Loss {loss:.4f} | TrainAcc {acc * 100:.2f}% | LR {current_lr:.5f}")

            if step % 50 == 0:
                for i, block in enumerate(model.blocks):
                    print(f"  [Stats] Block {i + 1} Entropy: {block.attn.last_entropy:.3f}")

        if step % EVAL_INTERVAL == 0 and step > 0:
            val_loss, val_acc = evaluate_metrics(model, val_data)
            print(f"\n>>> [EVAL PERFORMER] Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                try:
                    torch.save(model.state_dict(), MODEL_PATH)
                    print(f"✅ Model Saved (Best Acc: {best_acc * 100:.2f}%)")
                except Exception as e:
                    
                    print(f"⚠️ Model Save Failed: {e}")
            print("-" * 40)


if __name__ == "__main__":
    train()