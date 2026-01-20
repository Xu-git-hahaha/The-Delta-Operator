import torch
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import os


SAVE_PATH = "data/tiny_imagenet_64x64.pt"




def prepare_tiny_imagenet():
    if os.path.exists(SAVE_PATH):
        print(f"✅ 文件已存在: {SAVE_PATH}")
        return

    print("🚀 开始通过 Hugging Face 下载 Tiny ImageNet...")
    
    try:
        dataset = load_dataset("zh-plus/tiny-imagenet", split="train")
    except Exception as e:
        print(f"下载失败: {e}")
        return

    print(f"📦 数据集加载完成，共 {len(dataset)} 张图片")
    print(f"⚙️ 正在预处理: ToTensor -> Normalize([-1, 1])...")

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    all_tensors = []
    labels = []

    
    batch_size = 1000
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i: i + batch_size]
        imgs = batch['image']
        lbls = batch['label']  

        processed_imgs = []
        for img in imgs:
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            processed_imgs.append(transform(img))

        all_tensors.append(torch.stack(processed_imgs))
        labels.append(torch.tensor(lbls))

    print("💾 正在拼接并保存...")
    full_tensor = torch.cat(all_tensors, dim=0)
    full_labels = torch.cat(labels, dim=0)

    
    data_dict = {
        "images": full_tensor,
        "labels": full_labels
    }

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(data_dict, SAVE_PATH)

    print(f"✅ 成功! 数据已保存至 {SAVE_PATH}")
    print(f"📊 图片形状: {full_tensor.shape}, 标签形状: {full_labels.shape}")
    print(f"📁 文件大小: {os.path.getsize(SAVE_PATH) / (1024 ** 3):.2f} GB")


if __name__ == "__main__":
    prepare_tiny_imagenet()