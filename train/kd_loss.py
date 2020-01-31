import torch
from torch import nn, Tensor
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation loss as proposed by Hinton et al.
    """
    def __init__(self, distill_ratio: float, temperature: float):
        """
        Set distill_ratio and temperature parameters for KD loss calculation.
        This paper https://arxiv.org/pdf/1912.10850.pdf has a clean equation for the loss in Eq (1).
        Args:
            distill_ratio: scalar defining ratio of knowledge distillation to ground truth learning losses.
                (1-distill_ratio)*CE+(distill_ratio)*KL is used to split the two components.
                CE for cross-entropy with ground-truth labels, KL for KL divergence between student and teacher logits.
            temperature: temperature factor for increasing entropy in the KL divergence portion.
                The square of the temperature is multiplied to the KL portion for rescaling gradients.
                Applying temperature scales gradients by 1/T^2. Multiply with T^2 to restore the original scale.
        """
        super().__init__()
        assert 0 <= distill_ratio <= 1, 'Invalid value for distill_ratio.'
        assert 0 < temperature, 'Invalid value for temperature.'
        self.distill_ratio = distill_ratio
        self.temperature = temperature
        # Defining ratio factors.
        self.ce = 1 - distill_ratio  # Cross Entropy ratio.
        self.kl = distill_ratio * temperature * temperature  # KL Divergence ratio.

    def forward(self, student_logits: Tensor, teacher_logits: Tensor, targets: Tensor) -> (Tensor, dict):
        """
        Pytorch KL Divergence expects log-probabilities as inputs. Targets are true probabilities.
        Also, the reduction='mean' is incorrectly implemented and reduction='batchmean'
        is needed for the correct answer until the next major release (Pytorch 2.x).
        In D_KL(P||Q), P is target and Q is input.
        The order in the function might be somewhat confusing since targets come second, not first.
        Gradients are detached from teacher because teachers should not be trained during knowledge distillation.
        Args:
            student_logits: logit outputs from student model
            teacher_logits: logit outputs from teacher model
            targets: classification target labels as zero indexed integers, same as ordinary classification targets.

        Returns:
            The knowledge distillation loss as a single scalar tensor and the other losses in a dictionary.
            The other losses are returned so that they can be displayed on Tensorboard.

        """
        log_student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        with torch.no_grad():  # Small speedup by removing unnecessary gradient calculations.
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        cross_entropy_loss = F.cross_entropy(input=student_logits, target=targets)
        distillation_loss = F.kl_div(input=log_student_probs, target=teacher_probs.detach(), reduction='batchmean')
        kd_loss = self.ce * cross_entropy_loss + self.kl * distillation_loss
        return kd_loss, {'cross_entropy_loss': cross_entropy_loss, 'distillation_loss': distillation_loss}
