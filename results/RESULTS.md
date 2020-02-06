# Knowledge Distillation Results

##### The training settings for all models were identical and were as follows.

1. 400 epochs on the CIFAR10 dataset.
2. Linear warmup of learning rate for 5 epochs.
3. Learning rate decay by 10-fold after epochs 300, 350, 375.
4. Data augmentation by random horizontal flip, 4-pixel padding, 32 sized random cropping, then division by 255.
5. Batch size of 512.

##### The best performances of models on the test set are as follows.

1. Teacher (VGG16BN): 93.53%
2. Student (No KD): 88.31%
3. Student (Only KD): 88.28%
4. Student (distill_ratio=0.5, temperature=32): 89.76%


N.B. Reproducibility was not perfect, despite the reproducibility settings.


### Analysis
1. Using no knowledge distillation and using only knowledge distillation produces similar results.
2. Knowledge distillation improved performance in these settings by approximately 1.5%.
3. There seem to be several clusters of near-optimal settings for hyper-parameters.
