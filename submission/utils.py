"""공통 유틸리티: 데이터 로딩, 모델 생성, 옵티마이저/스케줄러 설정.

CIFAR-100 기준:
- Train 40k / Val 10k / Test 10k (seed=0 고정)
- ResNet-18, SGD momentum=0.9, wd=5e-4, lr=0.1 cosine annealing
- batch_size=128
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


# ---------------------------------------------------------------------------
# 시드 고정
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """재현성을 위한 랜덤 시드 고정."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 데이터 로더
# ---------------------------------------------------------------------------

def get_dataloaders(
    batch_size: int = 128, num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """CIFAR-100 데이터 로더 생성.

    50k train set을 40k train / 10k val로 분할 (seed=0).
    Train에만 augmentation (RandomCrop + RandomHorizontalFlip) 적용.
    """
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    # 학습용 augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    # 검증/테스트용 (augmentation 없음)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Augmented dataset에서 train 40k 추출
    train_full = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform,
    )
    train_dataset, _ = random_split(
        train_full, [40000, 10000],
        generator=torch.Generator().manual_seed(0),
    )

    # Non-augmented dataset에서 val 10k 추출 (같은 seed → 동일 분할)
    val_full = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=test_transform,
    )
    _, val_dataset = random_split(
        val_full, [40000, 10000],
        generator=torch.Generator().manual_seed(0),
    )

    # 테스트셋 10k
    test_dataset = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------

def get_model(device: torch.device, num_classes: int = 100) -> nn.Module:
    """ResNet-18 생성 (CIFAR-100용, num_classes=100)."""
    model = models.resnet18(num_classes=num_classes)
    return model.to(device)


# ---------------------------------------------------------------------------
# 옵티마이저 & 스케줄러
# ---------------------------------------------------------------------------

def get_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    T_max: int = 200,
) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
    """SGD + CosineAnnealingLR 생성.

    과제 스펙: SGD(lr=0.1, momentum=0.9, wd=5e-4), cosine annealing T_max=200.
    """
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# GPU-resident 빠른 평가 (val/test 전체를 GPU에 올려서 사용)
# ---------------------------------------------------------------------------

@torch.no_grad()
def fast_evaluate(model: nn.Module, images: torch.Tensor, targets: torch.Tensor,
                  batch_size: int = 2048) -> float:
    """GPU에 미리 올린 이미지/타겟으로 빠른 정확도 측정 (%).

    DataLoader 없이 텐서를 직접 mini-batch로 나눠서 평가.
    """
    model.eval()
    correct = 0
    with torch.cuda.amp.autocast():
        for i in range(0, len(images), batch_size):
            logits = model(images[i:i + batch_size])
            correct += (logits.argmax(1) == targets[i:i + batch_size]).sum().item()
    return correct / len(images) * 100.0


@torch.no_grad()
def fast_confusion_matrix(model: nn.Module, images: torch.Tensor, targets: torch.Tensor,
                          num_classes: int, batch_size: int = 2048) -> torch.Tensor:
    """GPU-resident 혼동행렬 계산 (논문 Eq.7).

    C_ij = 클래스 i 샘플에 대한 클래스 j의 평균 negative log-probability.
    """
    model.eval()
    C = torch.zeros(num_classes, num_classes, device=images.device)
    class_counts = torch.zeros(num_classes, device=images.device)
    with torch.cuda.amp.autocast():
        for i in range(0, len(images), batch_size):
            logits = model(images[i:i + batch_size])
            neg_log_probs = -F.log_softmax(logits.float(), dim=1)
            bt = targets[i:i + batch_size]
            C.index_add_(0, bt, neg_log_probs)
            class_counts.scatter_add_(0, bt, torch.ones(bt.size(0), device=images.device))
    nonzero = class_counts > 0
    C[nonzero] /= class_counts[nonzero].unsqueeze(1)
    return C
