import os
import torch
import bisect
import glob
import logging  
from torch.utils.data import Dataset

from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.utils import logging as hf_logging  
from DeltaLLM import Delta1BModel, DeltaConfig


os.environ["WANDB_DISABLED"] = "true"



OUTPUT_DIR = "checkpoints_delta_llm-1B"
DATA_DIR = "data_cache_llama"



class FileLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            
            logging.info(f"Step {state.global_step}: {logs}")



class LazyChunkDataset(Dataset):
    def __init__(self, data_dir, prefix="train_chunk"):
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.pt")))
        if not self.file_paths:
            raise FileNotFoundError(f"No {prefix}_*.pt files found in {data_dir}. Did you run prepare_data_llama.py?")

        print(f"📊 Found {len(self.file_paths)} chunks. Indexing metadata...")

        self.file_lengths = []
        self.cumulative_lengths = [0]
        self.cached_chunk = (None, None)

        for fp in self.file_paths:
            try:
                t = torch.load(fp, map_location="cpu")
                length = t.shape[0]
                self.file_lengths.append(length)
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
                del t
            except Exception as e:
                print(f"⚠️ Error reading {fp}: {e}")

        self.total_len = self.cumulative_lengths[-1]
        print(f"✅ Indexed {self.total_len} samples.")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        offset = idx - self.cumulative_lengths[file_idx]
        target_file = self.file_paths[file_idx]

        if self.cached_chunk[0] != target_file:
            self.cached_chunk = (target_file, torch.load(target_file, map_location="cpu"))

        sample = self.cached_chunk[1][offset]
        return {"input_ids": sample.long(), "labels": sample.long()}


def train():
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "training.log")

    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"))

    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, logging.StreamHandler()]
    )

    
    hf_logging.set_verbosity_info()
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()
    hf_logging.get_logger("transformers").addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"📝 Logging started. Logs will be saved to {log_file}")

    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 Using NVIDIA CUDA: {gpu_name} (VRAM: {vram:.2f} GB)")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("⚡ TF32 acceleration enabled.")

    elif torch.backends.mps.is_available():
        print("🚀 Using Apple MPS acceleration.")
    else:
        print("⚠️ No GPU detected. Using CPU (Slow).")

    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)

    
    print("🏗️ Initializing Delta-Nano (Llama-55k, Softmax Version) Model...")
    config = DeltaConfig(
        vocab_size=55296,
        hidden_size=2048,
        num_hidden_layers=28,
        num_attention_heads=16,
        intermediate_size=2048,
        use_cache=False
    )
    model = Delta1BModel(config)
    model.gradient_checkpointing = True
    print("🛡️ Gradient checkpointing enabled.")

    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"📊 Model Size: {params:.2f} Billion Parameters")

    
    print("📂 Loading Training Data...")
    train_dataset = LazyChunkDataset(DATA_DIR, prefix="train_chunk")

    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=False,

        num_train_epochs=1,
        optim="adamw_torch",

        
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,

        learning_rate=3e-4,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=10,

        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        dataloader_num_workers=8,

        logging_steps=1,
        save_strategy="steps",
        save_steps=50,

        eval_strategy="no",
        save_total_limit=3,
        load_best_model_at_end=False,
        save_safetensors=False,
        report_to="none",
        run_name="delta-nano-standard",
    )

    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        
        callbacks=[FileLoggerCallback()]
    )

    print(f"🔥 Starting Training with Standard AdamW!")

    resume = False
    if os.path.isdir(OUTPUT_DIR) and len(os.listdir(OUTPUT_DIR)) > 0:
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if len(checkpoints) > 0:
            resume = True
            print(f"🔄 Resuming from existing checkpoint: {checkpoints[-1]}...")

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    print("🎉 Training Finished!")


if __name__ == "__main__":
    train()