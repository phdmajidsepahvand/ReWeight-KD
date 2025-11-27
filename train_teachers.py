
import argparse
import os
import torch
import torch.nn as nn
from ecg_distill.data import create_dataloaders
from ecg_distill.models import ECGNet
from ecg_distill.utils import eval_accuracy, set_seed

def train_teacher(model, train_loader, test_loader, device, epochs=5, lr=1e-3):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        acc = eval_accuracy(model, test_loader, device)
        print(f"[Teacher] Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--num_teachers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, test_loader, meta = create_dataloaders(
        args.train_path, args.test_path, batch_size=args.batch_size
    )

    for i in range(args.num_teachers):
        print(f"=== Training Teacher {i+1} ===")
        teacher = ECGNet(meta["input_dim"], meta["num_classes"])
        teacher = train_teacher(teacher, train_loader, test_loader, device, epochs=args.epochs)
        torch.save(teacher.state_dict(), f"{args.out_dir}/teacher_{i+1}.pth")

if __name__ == "__main__":
    main()
