import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
import math


class StandardConfig(PretrainedConfig):
    model_type = "standard_1b"

    def __init__(
            self,
            vocab_size=55296,
            hidden_size=2048,
            intermediate_size=2048,
            num_hidden_layers=28,
            num_attention_heads=16,
            max_position_embeddings=2048,
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
        self.use_cache = use_cache



class StandardAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads

        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        
        
        

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=False)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)



class StandardMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        
        return self.down(F.silu(self.gate(x)) * self.up(x))


class StandardBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = StandardAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = StandardMLP(config)

    def forward(self, x, attention_mask=None):
        
        h = x + self.attn(self.ln1(x), attention_mask)
        out = h + self.mlp(self.ln2(h))
        return out


class StandardTransformer1BModel(PreTrainedModel, GenerationMixin):
    config_class = StandardConfig
    _supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.blocks = nn.ModuleList([StandardBlock(config) for _ in range(config.num_hidden_layers)])
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
        if isinstance(module, StandardTransformer1BModel):
            module.gradient_checkpointing = value

    def forward(self, input_ids, labels=None, **kwargs):
        B, T = input_ids.shape
        device = input_ids.device

        
        x = self.embed(input_ids)

        
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        x = x + self.pos_embed(pos_ids)

        
        
        
        mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
        
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

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


AutoConfig.register("standard_1b", StandardConfig)