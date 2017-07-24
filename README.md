# Theano Install Guide

http://blog.csdn.net/xierhacker/article/details/53035989

[Correct] .theanorc file should be:
 ======================================================================
[global]  
device=gpu  
floatX=float32 

[dnn.conv]
algo_bwd_filter = deterministic
algo_bwd_data = deterministic

[cuda]
root=/usr/local/cuda-8.0

[lib]
cnmem=0.3

[nvcc]
fastmath = True
optimizer_including=cudnn





cuDNN :/usr/bin/ld: 找不到 -lcudnn should follow:
 ======================================================================
NVIDIA provides a library for common neural network operations that especially speeds up Convolutional Neural Networks (CNNs). Again, it can be obtained from NVIDIA (after registering as a developer): https://developer.nvidia.com/cudnn

Note that it requires a reasonably modern GPU with Compute Capability 3.0 or higher; see NVIDIA’s list of CUDA GPUs.

To install it, copy the *.h files to /usr/local/cuda/include and the lib* files to /usr/local/cuda/lib64.

To check whether it is found by Theano, run the following command:

python -c "from theano.sandbox.cuda.dnn import dnn_available as d; print(d() or d.msg)"

