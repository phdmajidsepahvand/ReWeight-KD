
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

class TeacherEnsemble(nn.Module):
    def __init__(self, teachers):
        super().__init__()
        self.teachers = nn.ModuleList(teachers)

    @torch.no_grad()
    def forward(self, x):
        probs_list = []
        for t in self.teachers:
            logits = t(x)
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)

        probs_stack = torch.stack(probs_list, dim=0)  # [T, B, C]
        probs_mean = probs_stack.mean(dim=0)
        return probs_mean
