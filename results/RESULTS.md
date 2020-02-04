# Knowledge Distillation Results

##### The training settings for all models were identical and were as follows.

1. 400 epochs on th CIFAR10 dataset.
2. Linear warmup of learning rate for 5 epochs.
3. Learning rate decay by x0.1 after epochs 300, 350, 375.
4. Data augmentation by 4-pixel padding, 32 sized cropping, random horizontal flip, then division by 255.


##### The best performances of the models are as follows.

1. Teacher (VGG16BN): 93.74%
2. Student (No KD): 88.41%
3. Student (Only KD): 88.28%
4. Student (distill_ratio=0.5, temperature=32): 89.76%

