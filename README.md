Based on [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn)

# CPU faster-RCNN

1. Deleted all training scripts  
2. Wrote a new script for extracting all boxes from each image in a given folder
3. Compiled for CPUs (exclude all CUDA implementations). The original build files for CPU version are wrong, we corrected them.  
4. Wrote a `.sh` script. Tried to be user-friendly.  
5. The C-implemented layers are already compiled. Should work out-of-the-box.

Trained models can be downloaded from the [RELEASE page of faster-RCNN-gpu](https://github.com/BASFProject-2018spring/faster-RCNN-gpu/releases)

# How to use

This is a CPU implementation for platforms without CUDA support. The `run.py` inside tools folder has the same usage as the [GPU version](https://github.com/BASFProject-2018spring/faster-RCNN-gpu)
