# Knowledge-Distillation-Task

My implementation of "Distilling the Knowledge in a Neural Network".

This project is implemented with Pytorch 1.4 but will probably work on Pytorch 1.1 or above.

Python 3.7 was used with extensive use of type hinting and f-strings.

To run the project on the command line, first __*change current working directory to the project root*__.

Then use the following commands on the command line.

1. Train teacher: `python -m train.train_teacher (options)`
2. Train student: `python -m train.train_student (options)`
3. Grid Search: `python -m train.grid_search (options)`

Example usage:
1. `python -m train.train_teacher --gpu 0`
2. `python -m train.grid_search --gpu 0 --teacher_checkpoint ./records/Teacher/Trial 08  2020-01-30 14-22-32/checkpoints/checkpoint_040.tar`

Visualization on Tensorboard requires the following command:

`tensorboard --logdir ./path/to/logs`

#### Behavior

The outputs of experiments will not be uploaded to GitHub. 

Modify the .gitignore file to change this behavior. 

This is done because the file sizes are too large to upload.

#### Limitations

Only CIFAR10 is implemented for this project.

Also, the teacher and student models are fixed beforehand.

Data augmentation method, optimizer type, learning rate scheduling, etc.
cannot be altered via the command line with the current implementation.

Reproducibility is fixed on a single seed. 

Reproducibility settings also reduce speeds.

Moreover, this project assumes the use of a single GPU device for all models.
