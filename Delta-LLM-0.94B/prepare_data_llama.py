import os


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import gc
import time
import warnings

warnings.filterwarnings("ignore")


OUTPUT_DIR = "./data_cache_llama"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE_PREFIX = os.path.join(OUTPUT_DIR, "train_chunk")
VAL_FILE = os.path.join(OUTPUT_DIR, "val.pt")

TOKENIZER_NAME = "hfl/chinese-llama-2-7b"


TARGET_TOKENS = 500_000_000
CHUNK_SIZE = 100
MAX_LENGTH = 2048


RATIO = {"cn": 0.3, "en": 0.3, "code": 0.2, "sft": 0.2}


def setup_tokenizer():
    print(f"Loading Tokenizer: {TOKENIZER_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = 2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_chatml(response, instruction=None, data_type="generic"):
    if not response: return ""
    response = response.strip()
    if instruction:
        instruction = instruction.strip()
        return f"User: {instruction}\nAssistant: {response}\n"
    if len(response) < 5: return ""
    if data_type == "cn":
        return f"User: 请详细解释以下内容。\nAssistant: {response}\n"
    else:
        return f"User: Explain the following text.\nAssistant: {response}\n"



def get_stream_dataset(ratio_key):
    print(f"   -> Connecting stream for [{ratio_key}]...")
    ds = None
    if ratio_key == "cn":
        try:
            print("      Loading: pleisto/wikipedia-cn-20230720-filtered")
            ds = load_dataset("pleisto/wikipedia-cn-20230720-filtered", split="train", streaming=True)
        except Exception as e:
            print(f"      Wikipedia failed ({e}), switching to SkyPile...")
            ds = load_dataset("Skywork/SkyPile-150B", split="train", streaming=True)

    elif ratio_key == "en":
        print("      Loading: HuggingFaceFW/fineweb-edu")
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    elif ratio_key == "code":
        print("      Loading: iamtarun/python_code_instructions_18k_alpaca")
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)

    elif ratio_key == "sft":
        print("      Loading: YeungNLP/moss-003-sft-data")
        ds = load_dataset("YeungNLP/moss-003-sft-data", split="train", streaming=True)

    return ds


def process_and_save():
    tokenizer = setup_tokenizer()
    EOS_ID = tokenizer.eos_token_id

    print("\n📡 Connecting to Data Streams (Using Mirror)...")
    try:
        ds_cn = get_stream_dataset("cn")
        ds_en = get_stream_dataset("en")
        ds_code = get_stream_dataset("code")
        ds_sft = get_stream_dataset("sft")
    except Exception as e:
        print(f"\nInit Error: {e}")
        print("提示: 请检查网络，或稍后重试。镜像源已启用。")
        return

    iter_cn = iter(ds_cn)
    iter_en = iter(ds_en)
    iter_code = iter(ds_code)
    iter_sft = iter(ds_sft)

    token_buffer = []
    ready_samples = []
    val_samples = []
    chunk_idx = 0
    total_tokens_processed = 0

    
    pbar = tqdm(total=TARGET_TOKENS, desc="Packing Tokens", unit="tok")

    while total_tokens_processed < TARGET_TOKENS:
        try:
            batch_texts = []

            
            for _ in range(3):
                try:
                    item = next(iter_cn)
                    text = item.get('completion', item.get('text', ''))
                    if len(text) > 20: batch_texts.append(format_chatml(text, None, "cn"))
                except StopIteration:
                    pass

            
            for _ in range(3):
                try:
                    item = next(iter_en)
                    text = item.get('text', '')
                    if len(text) > 20: batch_texts.append(format_chatml(text, None, "en"))
                except StopIteration:
                    pass

            
            for _ in range(2):
                try:
                    item = next(iter_code)
                    prompt = item.get('instruction', '')
                    input_ctx = item.get('input', '')
                    code_res = item.get('output', '')
                    if input_ctx: prompt += f"\nInput Context: {input_ctx}"
                    if len(code_res) > 10: batch_texts.append(format_chatml(code_res, prompt, "code"))
                except StopIteration:
                    pass

            
            for _ in range(2):
                try:
                    item = next(iter_sft)
                    instruction = item.get('instruction', item.get('prompt', ''))
                    response = item.get('output', item.get('response', ''))
                    if len(instruction) > 2: batch_texts.append(format_chatml(response, instruction, "sft"))
                except StopIteration:
                    pass

            if not batch_texts: break

            
            for text in batch_texts:
                ids = tokenizer.encode(text, add_special_tokens=False)
                ids.append(EOS_ID)
                token_buffer.extend(ids)

            while len(token_buffer) >= MAX_LENGTH:
                chunk = token_buffer[:MAX_LENGTH]
                token_buffer = token_buffer[MAX_LENGTH:]
                tensor_chunk = torch.tensor(chunk, dtype=torch.int32)

                if len(val_samples) < 500 and np.random.rand() < 0.01:
                    val_samples.append(tensor_chunk)
                else:
                    ready_samples.append(tensor_chunk)

                total_tokens_processed += MAX_LENGTH
                pbar.update(MAX_LENGTH)

            if len(ready_samples) >= CHUNK_SIZE:
                save_path = f"{TRAIN_FILE_PREFIX}_{chunk_idx}.pt"
                torch.save(torch.stack(ready_samples), save_path)
                pbar.write(f"Saved Chunk {chunk_idx}")
                ready_samples = []
                chunk_idx += 1
                gc.collect()

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            pbar.write(f"⚠Error: {e}")
            time.sleep(1)

    pbar.close()
    if ready_samples:
        torch.save(torch.stack(ready_samples), f"{TRAIN_FILE_PREFIX}_{chunk_idx}.pt")
    if val_samples:
        torch.save(torch.stack(val_samples), VAL_FILE)

    print(f"\nDone! Data saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    process_and_save()