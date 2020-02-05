# Knowledge-Distillation-Task

My implementation of "Distilling the Knowledge in a Neural Network".

This project is implemented with Pytorch 1.4 but will probably work on Pytorch 1.1 or above.

Python 3.7 was used with extensive use of type hinting and f-strings.

#### Behavior

The outputs of experiments will not be uploaded to GitHub. 

Modify the .gitignore file to change this behavior. 

This is done because the file sizes are too large to upload.

#### Limitations

Only CIFAR10 is implemented for this project.

Also, the teacher and student models are fixed beforehand.

Data augmentation method, optimizer type, learning rate scheduling, etc.
cannot be altered via the command line with the current implementation.

Moreover, this project assumes the use of a single GPU device for all models.