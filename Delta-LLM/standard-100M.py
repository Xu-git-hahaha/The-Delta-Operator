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
MODEL_PATH = "baseline_standard_scheduler_dropout0.05_100M_wiki.pt"





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







class StandardAttention(nn.Module):
    """
    Standard Scaled Dot-Product Attention
    Replaces ProxyStudentTAttention for baseline comparison.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.scale = 1.0 / math.sqrt(self.d_head)
        self.last_entropy = 0.0

    def forward(self, x, mask=None):
        B, T, C = x.shape

        
        q = self.w_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        
        
        scores = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(scores, dim=-1)

        
        self.last_entropy = float(-(attn_probs * torch.log(attn_probs + 1e-9)).sum(-1).mean().detach())

        out = attn_probs @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.w_o(out)


class StandardSwiGLU(nn.Module):
    """
    Standard SwiGLU FFN
    Replaces DeltaSwiGLU.
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


class StandardBlock(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = StandardAttention(d_model, heads)
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

        
        
        self.blocks = nn.Sequential(*[StandardBlock(D_MODEL, N_HEADS) for _ in range(N_LAYERS)])

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
    Paper Analysis Utility:
    Calculates parameter breakdown for BaselineGPT for fair comparison table.
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
    print("📝 PAPER ANALYSIS REPORT: BASELINE MODEL ARCHITECTURE")
    print("=" * 60)
    print(f"Total Parameters         : {total_params / 1e6:.2f} M")
    print("-" * 30)
    print(f"Embedding Params         : {embedding_params / 1e6:.2f} M")
    print(f"Head Params (Final Proj) : {head_params / 1e6:.2f} M")
    print("-" * 30)
    print(f"Backbone Params (Computation Core) : {backbone_params / 1e6:.2f} M")
    print(f"  └─ Standard Params (Mul/Add)     : {backbone_params / 1e6:.2f} M")
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

    print("Starting Standard Baseline Training...")

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
            print(f"\n>>> [EVAL BASELINE] Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                try:
                    torch.save(model.state_dict(), MODEL_PATH)
                    print(f"✅ Model Saved (Best Acc: {best_acc * 100:.2f}%)")
                except:
                    pass
            print("-" * 40)


if __name__ == "__main__":
    train()