import torch
from torch import nn, Tensor
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation loss as proposed by Hinton et al.
    """
    def __init__(self, alpha: float, temperature: float):
        """
        Set alpha and temperature parameters for KD loss calculation.
        This paper https://arxiv.org/pdf/1912.10850.pdf has a clean equation for the loss in Eq (1).
        Args:
            alpha: scalar defining ratio of cross-entropy to KL divergence losses.
                (1-alpha)*CE+(alpha)*KL is used to split the two components.
            temperature: temperature factor for increasing entropy in the KL divergence portion.
        """
        super().__init__()
        assert 0 <= alpha <= 1, 'Invalid value for alpha.'
        assert 0 < temperature, 'Invalid value for temperature.'
        self.alpha = alpha
        self.temperature = temperature
        # Defining ratio factors.
        self.ce = 1 - alpha  # Cross Entropy ratio.
        self.kl = alpha * temperature * temperature  # KL Divergence ratio.

    def forward(self, student_logits: Tensor, teacher_logits: Tensor, targets: Tensor) -> (Tensor, dict):
        """
        Pytorch KL Divergence expects log-probabilities as inputs. Targets are true probabilities.
        Also, the reduction='mean' is incorrectly implemented and reduction='batchmean'
        is needed for the correct answer until the next major release (Pytorch 2.x).
        In D_KL(P||Q), P is target and Q is input.
        The order in the function might be somewhat confusing because of this.
        Gradients are detached from teacher just in case they haven't been already.
        Args:
            student_logits: logit outputs from student model
            teacher_logits: logit outputs from teacher model
            targets: classification target labels as zero indexed integers

        Returns:
            The knowledge distillation loss as a single scalar tensor and the other losses in a dictionary.
            The other losses are returned so that they can be displayed on Tensorboard.

        """
        log_student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        with torch.no_grad:  # Small speedup by removing unnecessary gradient calculations.
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(input=log_student_probs, target=teacher_probs.detach(), reduction='batchmean')
        cross_entropy_loss = F.cross_entropy(input=student_logits, target=targets)
        kd_loss = self.ce * cross_entropy_loss + self.kl * distillation_loss
        return kd_loss, {'cross_entropy_loss': cross_entropy_loss, 'distillation_loss': distillation_loss}
