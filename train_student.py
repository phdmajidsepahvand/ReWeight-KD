
import argparse
import os
import torch
import torch.nn.functional as F

from ecg_distill.data import create_dataloaders
from ecg_distill.models import ECGNet, TeacherEnsemble
from ecg_distill.distillation import compute_sample_weights_from_distribution, distillation_loss_weighted
from ecg_distill.utils import eval_accuracy, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--teachers_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scale", type=float, default=30.0)
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, meta = create_dataloaders(
        args.train_path, args.test_path, batch_size=args.batch_size
    )

    # Load teachers
    teacher_files = sorted(
        f for f in os.listdir(args.teachers_dir) if f.startswith("teacher_")
    )
    teachers = []
    for f in teacher_files:
        model = ECGNet(meta["input_dim"], meta["num_classes"]).to(device)
        model.load_state_dict(torch.load(os.path.join(args.teachers_dir, f), map_location=device))
        model.eval()
        teachers.append(model)

    ensemble = TeacherEnsemble(teachers).to(device)

    # Student model
    student = ECGNet(meta["input_dim"], meta["num_classes"]).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        student.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                teacher_probs = ensemble(x)

            logits = student(x)

            weights = compute_sample_weights_from_distribution(teacher_probs, scale=args.scale)
            loss_kd = distillation_loss_weighted(logits, teacher_probs, weights)
            loss_ce = F.cross_entropy(logits, y)
            loss = loss_ce + loss_kd

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        acc = eval_accuracy(student, test_loader, device)
        print(f"[Student] Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f} acc={acc:.4f}")

if __name__ == "__main__":
    main()
