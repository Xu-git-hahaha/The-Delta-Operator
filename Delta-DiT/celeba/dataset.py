import torch
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


IMAGE_SIZE = 64
BATCH_SIZE = 1000  
SAVE_PATH = "data/celeba_64x64.pt"




def prepare_celeba():
    if os.path.exists(SAVE_PATH):
        print(f"✅ 文件已存在: {SAVE_PATH}，直接加载使用即可。")
        return

    print("🚀 开始通过 Hugging Face 下载 CelebA (Aligned)...")
    
    
    try:
        dataset = load_dataset("nielsr/celeba-faces", split="train")
    except Exception as e:
        print(f"Hugging Face 下载失败，请检查网络 (可能需要梯子): {e}")
        return

    print(f"📦 数据集加载完成，共 {len(dataset)} 张图片")
    print(f"⚙️ 正在预处理: Resize({IMAGE_SIZE}) -> ToTensor -> Normalize([-1, 1])...")

    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE),  
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])

    
    def process_batch(batch):
        
        pixel_values = [transform(img.convert("RGB")) for img in batch['image']]
        return torch.stack(pixel_values)

    
    all_tensors = []

    
    
    total = len(dataset)
    for i in tqdm(range(0, total, BATCH_SIZE)):
        end = min(i + BATCH_SIZE, total)
        batch_imgs = dataset[i:end]
        processed = process_batch(batch_imgs)
        all_tensors.append(processed)

    print("💾 正在拼接并保存为 .pt 文件 (这可能需要几分钟)...")
    full_tensor = torch.cat(all_tensors, dim=0)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(full_tensor, SAVE_PATH)

    print(f"✅ 成功! 数据已保存至 {SAVE_PATH}")
    print(f"📊 Tensor 形状: {full_tensor.shape}")
    print(f"   (N, C, H, W) = ({full_tensor.shape[0]}, 3, 64, 64)")
    print(f"📁 文件大小: {os.path.getsize(SAVE_PATH) / (1024 ** 3):.2f} GB")


if __name__ == "__main__":
    prepare_celeba()