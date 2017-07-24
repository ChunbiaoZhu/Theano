# Theano
Install
# Guide

#[Correct] .theanorc file should be:
http://blog.csdn.net/xierhacker/article/details/53035989

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
