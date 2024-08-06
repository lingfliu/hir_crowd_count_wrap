# Another wrap demo of ucs_alg_node

# The Algorithm
This algorithm estimate the crowd density pixel-by-pixel using the vgg model.

- Input: images downloaded from the minio server's ucs-alg bucket
- Output: a crowd density matrix of the same size as the input image, saved as file and uploaded to the minio server

By default, the model is fixed, thus the model selection will not take effects.

# Dependencies
- ucs_alg_node-0.1.6 or above
- numpy
- torch
- torchvision
- pillow

# Usage
see [main.py](main.py)