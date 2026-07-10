"""
training_loop.py — 完整的 compiled 训练循环示例

展示 torch.compile 在完整训练流程中的使用，包括：
- 模型编译
- 混合精度训练 (AMP)
- 梯度缩放 (GradScaler)
- 梯度累积
- 学习率调度
- 多编译模式对比

用法:
    python training_loop.py                    # 默认配置训练
    python training_loop.py --mode reduce-overhead  # 使用 reduce-overhead 模式
    python training_loop.py --amp             # 启用混合精度
    python training_loop.py --benchmark       # 运行模式对比
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import time
import os
import sys

# 将项目根目录加入 path，方便导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.benchmark_utils import timing_median, BenchmarkRunner


# ============================================================
# 模型定义
# ============================================================


class ResNetLikeBlock(nn.Module):
    """简化的 ResNet-like 模块，用于演示训练循环。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class SmallResNet(nn.Module):
    """用于训练演示的小型 ResNet。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResNetLikeBlock(64, 64, stride=1)
        self.layer2 = ResNetLikeBlock(64, 128, stride=2)
        self.layer3 = ResNetLikeBlock(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ============================================================
# 训练循环
# ============================================================


# --- docs: trainer ---


class CompiledTrainer:
    """封装了 compiled 训练循环的 Trainer。"""

    def __init__(
        self,
        model: nn.Module,
        compile_mode: str = "default",
        use_amp: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.use_amp = use_amp
        self.compile_mode = compile_mode

        # 编译模型
        if compile_mode == "eager":
            self.model = model.cuda()
        else:
            self.model = torch.compile(model, mode=compile_mode).cuda()

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个 epoch，返回平均 loss。"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()

            # 前向传播 — 使用 AMP 上下文
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                output = self.model(x)
                loss = self.criterion(output, y)

            # 反向传播 — 通过 scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

        self.scheduler.step()
        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> tuple:
        """评估模型，返回 (平均 loss, 准确率)。"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                output = self.model(x)
                loss = nn.CrossEntropyLoss()(output, y)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        return total_loss / len(dataloader), correct / total


# --- docs: end ---


# ============================================================
# 合成数据生成
# ============================================================


def create_synthetic_dataset(
    num_samples: int = 1024,
    image_size: int = 32,
    num_classes: int = 10,
    batch_size: int = 64,
):
    """生成合成图像数据用于演示。"""
    x = torch.randn(num_samples, 3, image_size, image_size)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# ============================================================
# 梯度累积演示
# ============================================================


def train_with_gradient_accumulation(
    model: nn.Module,
    dataloader: DataLoader,
    accumulation_steps: int = 4,
    compile_mode: str = "default",
    use_amp: bool = False,
):
    """展示梯度累积与 torch.compile 的配合。

    梯度累积通过累积多个 micro-batch 的梯度来模拟更大的 batch size。
    每个 micro-batch 的 forward/backward 都是独立的编译调用，
    torch.compile 不需要为此做特殊处理。
    """
    compiled_model = torch.compile(model, mode=compile_mode).cuda()
    optimizer = optim.SGD(compiled_model.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    for i, (x, y) in enumerate(dataloader):
        x, y = x.cuda(), y.cuda()

        with torch.amp.autocast("cuda", enabled=use_amp):
            output = compiled_model(x)
            # 除以 accumulation_steps 使得最终梯度等价于更大的 batch
            loss = criterion(output, y) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


# ============================================================
# 主函数
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="torch.compile 训练循环示例")
    parser.add_argument(
        "--mode",
        default="default",
        choices=["eager", "default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--amp", action="store_true", help="启用混合精度训练")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--benchmark", action="store_true", help="运行模式对比 benchmark"
    )
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
        return

    print(f"训练配置: mode={args.mode}, amp={args.amp}, epochs={args.epochs}")
    print("-" * 50)

    model = SmallResNet(num_classes=10)
    trainer = CompiledTrainer(
        model,
        compile_mode=args.mode,
        use_amp=args.amp,
    )

    dataloader = create_synthetic_dataset(
        num_samples=2048,
        batch_size=args.batch_size,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch:2d}/{args.epochs}  |  平均 Loss: {train_loss:.4f}")

    print("-" * 50)
    print("训练完成！")


def run_benchmark():
    """对比不同编译模式的训练吞吐量。"""
    print("运行训练 benchmark...\n")

    model = SmallResNet(num_classes=10)
    dataloader = create_synthetic_dataset(num_samples=512, batch_size=32)

    modes = ["eager", "default", "reduce-overhead"]
    results = []

    for mode in modes:
        trainer = CompiledTrainer(
            model,
            compile_mode=mode,
            use_amp=False,
        )

        # 热身后测量一个 epoch 的时间
        trainer.train_epoch(dataloader)  # 热身（包含编译开销）

        elapsed = timing_median(
            trainer.train_epoch,
            dataloader,
            n_warmup=1,
            n_iter=3,
            sync=True,
        )

        results.append((mode, elapsed))
        print(f"  {mode:<20s} {elapsed:.2f} ms/epoch")

    # 计算加速比
    eager_time = (
        results[0][1]
        if results[0][0] == "eager"
        else next(t for m, t in results if m == "eager")
    )
    print("\n加速比:")
    for mode, elapsed in results:
        if mode != "eager":
            print(f"  {mode:<20s} {eager_time / elapsed:.2f}x")


if __name__ == "__main__":
    main()
