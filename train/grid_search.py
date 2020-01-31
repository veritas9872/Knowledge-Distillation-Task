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
        batch_size=256,
        num_workers=2,
        lr=0.1,
        gpu=1
    )
    temperatures = [1, 2, 4, 8, 16, 32]
    distill_ratios = [0.99, 0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]

    for temp in temperatures:
        for dist in distill_ratios:
            options['temperature'] = temp
            options['distill_ratio'] = dist
            opt = knowledge_distillation_options(**options).parse_args()
            main(opt)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True  # Increase speed if input sizes are the same.
    grid_search()
