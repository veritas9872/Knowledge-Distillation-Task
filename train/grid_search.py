"""
Code for applying grid search to find the best parameters for knowledge distillation.
The distillation ratio and temperature parameters are being tuned in this search.
"""
import torch

from train.distill_knowledge import main
from utils.options import knowledge_distillation_options


def grid_search():
    options = dict(
        teacher_checkpoint='../records/Teacher/Trial 08  2020-01-30 14-22-32/checkpoints/checkpoint_040.tar',
        train_method='Search',
        num_epochs=400,
        batch_size=512,
        num_workers=4,
        lr=0.1,
        gpu=1
    )
    temperatures = [1, 2, 4, 8, 16, 32, 64]
    distill_ratios = [1., 0.99, 0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.]

    for temp in temperatures:
        for dist in distill_ratios:
            options['temperature'] = temp
            options['distill_ratio'] = dist
            opt = knowledge_distillation_options(**options).parse_args()
            # Reproducibility settings. Seeding must be repeated at the start of every run.
            torch.random.manual_seed(9872)
            main(opt)


if __name__ == '__main__':
    # Reproducibility settings.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    grid_search()
