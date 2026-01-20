import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
import math


class DeltaConfig(PretrainedConfig):
    model_type = "delta_1b"

    def __init__(
            self,
            vocab_size=55296,
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=2048,
            Delta_scale=1.0,
            dir_scale=0.5,
            use_cache=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.Delta_scale = Delta_scale
        self.dir_scale = dir_scale
        self.use_cache = use_cache



class ProxyDeltaLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.gain = nn.Parameter(torch.ones(1, 1, out_dim) * 2.0)

    def forward(self, x):
        x_act = torch.tanh(x)
        w_act = torch.tanh(self.weight)

        B, T, D = x_act.shape
        x_flat = x_act.reshape(-1, D)

        x_sq = x_flat.pow(2).sum(dim=1, keepdim=True)
        w_sq = w_act.pow(2).sum(dim=1).unsqueeze(0)

        if self.out_dim > 8192:
            dot = self._chunked_matmul(x_flat, w_act.t())
        else:
            dot = torch.matmul(x_flat, w_act.t())

        l2_sq = (x_sq + w_sq - 2 * dot).clamp(min=0.0)
        l2_sq_mean = l2_sq / D

        out = 1.0 - l2_sq_mean
        out = out.view(B, T, self.out_dim)
        return (out - 0.5) * self.gain

    def _chunked_matmul(self, a, b):
        out_list = []
        chunk_size = 4096
        for i in range(0, b.shape[1], chunk_size):
            chunk = b[:, i: i + chunk_size]
            out_list.append(torch.matmul(a, chunk))
        return torch.cat(out_list, dim=1)



class DeltaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = ProxyDeltaLinear(config.hidden_size, config.hidden_size)
        self.k_proj = ProxyDeltaLinear(config.hidden_size, config.hidden_size)
        self.v_proj = ProxyDeltaLinear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim

        q = torch.tanh(self.q_proj(x)).view(B, T, H, D).transpose(1, 2)
        k = torch.tanh(self.k_proj(x)).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        q_sq = q.pow(2).sum(dim=-1, keepdim=True)
        k_sq = k.pow(2).sum(dim=-1, keepdim=True).transpose(-2, -1)
        dot = torch.matmul(q, k.transpose(-2, -1))

        dist_sq = (q_sq + k_sq - 2 * dot).clamp(min=0.0)

        
        logits = -dist_sq / (self.scale * 10.0)

        if attention_mask is not None:
            logits = logits.masked_fill(attention_mask == 0, float('-inf'))

        attn_probs = F.softmax(logits, dim=-1)

        out = attn_probs @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)


class DeltaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = DeltaAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = DeltaMLP(config)

        shift_pattern = [1, 2, 4]
        self.shift = shift_pattern[layer_idx % 3]
        self.dir_w = nn.Parameter(torch.randn(config.hidden_size) * 0.1)
        self.dir_scale = config.dir_scale

    def forward(self, x, attention_mask=None):
        h = x + self.attn(self.ln1(x), attention_mask)
        norm_h = self.ln2(h)
        ffn_out = self.mlp(norm_h)
        padded = F.pad(norm_h, (0, 0, self.shift, 0))[:, :-self.shift, :]
        dir_out = torch.abs(padded - torch.sigmoid(self.dir_w))
        out = h + ffn_out + dir_out * self.dir_scale
        return out


class DeltaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = ProxyDeltaLinear(config.hidden_size, config.intermediate_size)
        self.up = ProxyDeltaLinear(config.hidden_size, config.intermediate_size)
        self.down = ProxyDeltaLinear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Delta1BModel(PreTrainedModel, GenerationMixin):
    config_class = DeltaConfig

    
    _supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([DeltaBlock(config, i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.gradient_checkpointing = False

    
    @property
    def supports_gradient_checkpointing(self):
        return True

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Delta1BModel):
            module.gradient_checkpointing = value

    def forward(self, input_ids, labels=None, **kwargs):
        B, T = input_ids.shape
        x = self.embed(input_ids)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)

        if self.gradient_checkpointing and self.training:
            x.requires_grad_(True)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


AutoConfig.register("delta_1b", DeltaConfig)