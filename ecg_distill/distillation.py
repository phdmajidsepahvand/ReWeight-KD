
import torch
import torch.nn.functional as F

def compute_sample_weights_from_distribution(probs, scale=30.0):
    """
    Dirichlet-based sample weighting.
    alpha_k = p_k * scale
    alpha0 = sum(alpha_k)
    weight = alpha0 / (alpha0 + C)
    """
    alpha = probs * scale
    alpha0 = alpha.sum(dim=-1)
    C = probs.size(1)
    weights = alpha0 / (alpha0 + C)
    return weights.clamp(0.0, 1.0)

def distillation_loss_weighted(student_logits, teacher_probs, sample_weights, T=1.0):
    log_p_student = F.log_softmax(student_logits / T, dim=-1)
    p_teacher = teacher_probs

    kl = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=-1)
    return (kl * sample_weights).mean()
