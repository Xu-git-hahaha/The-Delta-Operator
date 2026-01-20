import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np




BATCH_SIZE = 128
EPOCHS = 300  
LR = 3e-3  
WEIGHT_DECAY = 0.05  
MIXUP_ALPHA = 1.0  
SAVE_PATH = "Delta_visnet_clean_best.pt"



def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    try:
        if torch.backends.mps.is_available(): return torch.device("mps")
    except:
        pass
    return torch.device("cpu")


DEVICE = get_device()
print(f"🔥 Delta-VisNet Clean Version Launching on: {DEVICE}")







class DeltaSpatialFused(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.ones(1, 1, 1, dim) * 0.5)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        
        kernel = torch.zeros(4 * dim, 1, 3, 3)

        
        kernel[:, 0, 1, 1] = 1.0
        
        kernel[0 * dim: 1 * dim, 0, 0, 1] = -1.0
        
        kernel[1 * dim: 2 * dim, 0, 2, 1] = -1.0
        
        kernel[2 * dim: 3 * dim, 0, 1, 0] = -1.0
        
        kernel[3 * dim: 4 * dim, 0, 1, 2] = -1.0

        
        self.register_buffer('diff_kernel', kernel)

    def forward(self, x):
        
        x_in = x.permute(0, 3, 1, 2)

        
        diff_all = F.conv2d(x_in, self.diff_kernel, padding=1, groups=self.dim)

        
        B, C4, H, W = diff_all.shape
        diff_split = diff_all.view(B, 4, self.dim, H, W)

        
        feat_edge = torch.abs(diff_split).sum(dim=1)  
        feat_edge = feat_edge.permute(0, 2, 3, 1)  

        
        global_potential = self.global_pool(x_in).permute(0, 2, 3, 1)

        return feat_edge * self.alpha + global_potential * (1 - self.alpha)



class DeltaChannel(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        self.hidden_dim = int(dim * expansion)

        
        self.proto_in = nn.Parameter(torch.randn(dim, self.hidden_dim) * 0.02)
        self.proto_out = nn.Parameter(torch.randn(self.hidden_dim, dim) * 0.02)

        
        self.scale_in = nn.Parameter(torch.ones(1) * 10.0)
        self.scale_out = nn.Parameter(torch.ones(1) * 10.0)

    def forward(self, x):
        B, H, W, C = x.shape
        x_flat = x.reshape(-1, C)  

        
        x_norm = F.normalize(x_flat, p=2, dim=1)
        w_in_norm = F.normalize(self.proto_in, p=2, dim=0)

        
        sim_in = torch.mm(x_norm, w_in_norm)
        hidden = F.gelu(sim_in * self.scale_in)

        
        h_norm = F.normalize(hidden, p=2, dim=1)
        w_out_norm = F.normalize(self.proto_out, p=2, dim=0)

        sim_out = torch.mm(h_norm, w_out_norm)
        out = sim_out * self.scale_out

        return out.reshape(B, H, W, C)



class DeltaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.spatial = DeltaSpatialFused(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.channel = DeltaChannel(dim, expansion=2)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-4)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.spatial(x)
        x = shortcut + self.gamma * x

        shortcut = x
        x = self.norm2(x)
        x = self.channel(x)
        x = shortcut + self.gamma * x
        return x



class DeltaVisNet(nn.Module):
    def __init__(self, num_classes=100, dims=[96, 192, 384], depths=[3, 3, 6]):
        super().__init__()
        print(f"🏗️ Building Model with Dims: {dims}, Depths: {depths}")

        
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 3, 1, 1),
            nn.BatchNorm2d(dims[0]),
            nn.GELU()
        )

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(len(dims)):
            stage = nn.Sequential(*[DeltaBlock(dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)
            if i < len(dims) - 1:
                
                self.downsamples.append(nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1),
                    nn.BatchNorm2d(dims[i + 1])
                ))

        self.norm = nn.LayerNorm(dims[-1])

        
        self.head_proto = nn.Parameter(torch.randn(dims[-1], num_classes) * 0.02)
        self.head_scale = nn.Parameter(torch.ones(1) * 15.0)

    def forward(self, x):
        x = self.stem(x)
        x = x.permute(0, 2, 3, 1).contiguous()

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                x = x.permute(0, 3, 1, 2).contiguous()
                x = self.downsamples[i](x)
                x = x.permute(0, 2, 3, 1).contiguous()

        x = self.norm(x)
        x = x.mean(dim=[1, 2])  

        
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.head_proto, p=2, dim=0)
        logits = torch.mm(x_norm, w_norm) * self.head_scale
        return logits





def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)





def train():
    
    print("📦 Loading CIFAR-100...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                              pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                             pin_memory=True)

    
    model = DeltaVisNet(
        num_classes=100,
        dims=[96, 192, 384],
        depths=[3, 3, 6]
    ).to(DEVICE)

    params = sum(p.numel() for p in model.parameters())
    print(f"🌊 Model Params: {params / 1e6:.2f}M")

    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(trainloader), epochs=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_acc = 0.0

    print("🏎️ Starting Delta-VisNet Training...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        
        pbar = tqdm(trainloader, desc=f"Ep {epoch + 1}/{EPOCHS}", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, MIXUP_ALPHA, DEVICE)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                loss.backward()
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f"{train_loss / (pbar.n + 1):.3f}"})

        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total

        
        if acc > best_acc:
            best_acc = acc
            print(f"🔥 New Best Acc: {best_acc:.2f}% | Saving to {SAVE_PATH}...")
            torch.save(model.state_dict(), SAVE_PATH)

        print(f"⏰ Ep {epoch + 1} | Val Acc: {acc:.2f}% (Best: {best_acc:.2f}%)")


if __name__ == '__main__':
    train()