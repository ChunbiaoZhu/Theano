# Theano Install Guide

http://blog.csdn.net/xierhacker/article/details/53035989
 ======================================================================
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




cuDNN :/usr/bin/ld: 找不到 -lcudnn 
ImportError: cuDNN not available: Can not compile with cuDNN. 
should follow:
 ======================================================================
NVIDIA provides a library for common neural network operations that especially speeds up Convolutional Neural Networks (CNNs). Again, it can be obtained from NVIDIA (after registering as a developer): https://developer.nvidia.com/cudnn

Note that it requires a reasonably modern GPU with Compute Capability 3.0 or higher; see NVIDIA’s list of CUDA GPUs.

To install it, copy the *.h files to /usr/local/cuda/include and the lib* files to /usr/local/cuda/lib64.

To check whether it is found by Theano, run the following command:

python -c "from theano.sandbox.cuda.dnn import dnn_available as d; print(d() or d.msg)"

ImportError: No module named cv2
 ======================================================================
 pip install opencv-python

 
 Ubuntu14.04和16.04官方默认更新源sources.list和第三方源推荐（干货！）
  ======================================================================
  http://www.cnblogs.com/zlslch/p/6860229.html
  
  Caffe 
    ======================================================================
  http://blog.csdn.net/zouyu1746430162/article/details/54095807
  
  http://blog.csdn.net/xierhacker/article/details/53035989
  
Caffe Compile Guide
=============
I think the reason is you use make to compile, which makes caffe's python port only find libraries in this catalog. Maybe you use cmaketo compile and it could work.
    make clean
    cd caffe-master
    mkdir build
    cd build
    cmake ..
    make all -j8




 

Some Problems for Deep Learning with Theano && Lasagne framework
=====================================================================
http://wzmlj.com/wangxiaocvpr/p/6605258.html
 

 

1. theano.function

　　 output = input ** 2 

　　 f = theano.function([input], output)

　　 print(f(3)) 

　　>> the output is: 3^2 = 9.

　　

 

2.  verbose = 1 or 0, does it have any difference ? 

　   some deep learning framework such as torch or theano, use this flag to control whether shown some training information on the terminal. 

　   such as: hist = Deep_Q_Network.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, verbose=1)

　　It will display on the terminal: 　　loss: ~~~

　　 　

　　

3. how to use specific GPU with Theano framework ? 

　　-->> device=gpu{0, 1, ...}  

　　the answer from: http://www.cnblogs.com/shouhuxianjian/p/4590224.html

　　

 

 

 4. ValueError: GpuJoin: Wrong inputs for input 2 related to inputs 0.!

0%| | 0/301 [00:00<?, ?it/s]
Traceback (most recent call last):
File "wangxiao-02-train.py", line 187, in <module>
train()
File "wangxiao-02-train.py", line 171, in train
salgan_batch_iterator(model, train_data, None)
File "wangxiao-02-train.py", line 108, in salgan_batch_iterator
G_obj, D_obj, G_cost = model.D_trainFunction(batch_input,batch_output,batch_patch) #add batch_patch ???
File "/usr/local/lib/python2.7/dist-packages/theano/compile/function_module.py", line 871, in __call__
storage_map=getattr(self.fn, 'storage_map', None))
File "/usr/local/lib/python2.7/dist-packages/theano/gof/link.py", line 314, in raise_with_op
reraise(exc_type, exc_value, exc_trace)
File "/usr/local/lib/python2.7/dist-packages/theano/compile/function_module.py", line 859, in __call__
outputs = self.fn()
ValueError: GpuJoin: Wrong inputs for input 2 related to inputs 0.!
Apply node that caused the error: GpuJoin(TensorConstant{1}, GpuElemwise{Composite{scalar_sigmoid((i0 + i1))},no_inplace}.0, GpuFromHost.0)
Toposort index: 955
Inputs types: [TensorType(int8, scalar), CudaNdarrayType(float32, (False, True, False, False)), CudaNdarrayType(float32, 4D)]
Inputs shapes: [(), (32, 1, 192, 256), (32, 3, 224, 224)]
Inputs strides: [(), (49152, 0, 256, 1), (150528, 50176, 224, 1)]
Inputs values: [array(1, dtype=int8), 'not shown', 'not shown']
Outputs clients: [[GpuContiguous(GpuJoin.0)]]

 

HINT: Re-running with most Theano optimization disabled could give you a back-trace of when this node was created. This can be done with by setting the Theano flag 'optimizer=fast_compile'. If that does not work, Theano optimizations can be disabled with 'optimizer=None'.
HINT: Use the Theano flag 'exception_verbosity=high' for a debugprint and storage map footprint of this apply node.

what's going on ???

 

==>> Because I input wrong image size: as shown in blue ! (32, 1, 192, 256), (32, 3, 224, 224)

 

 

5. Error when tring to find the memory information on the GPU: an illegal memory access was encountered

0%| | 0/301 [00:00<?, ?it/s]Error when tring to find the memory information on the GPU: an illegal memory access was encountered
Error freeing device pointer 0x10372a00000 (an illegal memory access was encountered). Driver report 0 bytes free and 0 bytes total
CudaNdarray_uninit: error freeing self->devdata. (self=0x7f19d0544830, self->devata=0x10372a00000)
Error when trying to find the memory information on the GPU: an illegal memory access was encountered
Error allocating 422125568 bytes of device memory (an illegal memory access was encountered). Driver report 0 bytes free and 0 bytes total
Segmentation fault 

==>>

 

 6.  some explnations about Jacobian matrix, Hessian matrix : 

　　from: http://blog.csdn.net/lanchunhui/article/details/50234117

 

 

 

7. some basic tutorials from theano Document v0.9:

　　(1). basic functin definition:

　　　　>> import numpy as np 

　　　　>> import theano.tensor as T

　　　　>> from theano import function 

　　　　>> x = T.dscalar('x')　　-- double style    the type we assign to “0-dimensional arrays (scalar) of doubles (d)”.

　　　　>> y = T.dscalar('y')

　　　　>> z = x + y

　　　　>> f = function([x, y], z)

 

　　when test, just input f(2, 3), then it will return z = 2+3 = array(5.0). We can also use the pp function to pretty-print out the computation associdated to z.

　　==>> print(pp(z)) 　　==>> (x+y).  

 

　　Also, we can add two matrices:  x = T.dmatrix('x')  

　　　　

　　(2). Derivatives in Theano 

　　　　<1>. Computing Gradients: using T.grad  

　　　　

　　　　<2>. Computing the Jacobian 

　　　　this blog provided us a good tutorial about this definiton: http://blog.csdn.net/lanchunhui/article/details/50234117 

　　　　

　　　　

　　

　　

8. set the GPU mode in theano framework:

　　vim ~/.theanorc 

　　and add these lines into this file:

　　[global]

　　device = gpu

　　floatX=float32

　　[nvcc]

　　flags=--machine=64

　　[lib] cnmem=100 　　

 

==>> then test these codes to see whether it shown success ?

from theano import function, config, shared, tensor
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')

　　

 

 

9. No Module named cv2 :

　　sudo apt-get install Python-OpenCV 

　　otherwise, you can try: 

　　　　pip install --upgrade setuptools
　　　　pip install numpy Matplotlib
　　　　pip install opencv-python

 

 

10. The following error happened while compiling the node': 

 

mod.cu(305): warning: subscript out of range
mod.cu(313): warning: subscript out of range
mod.cu(317): error: identifier "cudnnSetConvolutionNdDescriptor_v3" is undefined
1 error detected in the compilation of "/tmp/tmpxft_0000053a_00000000-9_mod.cpp1.ii".

 

['nvcc', '-shared', '-O3', '-arch=sm_61', '-m64', '-Xcompiler', '-fno-math-errno,-Wno-unused-label,-Wno-unused-variable,-Wno-write-strings,--machine=64,-DCUDA_NDARRAY_CUH=c72d035fdf91890f3b36710688069b2e,-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION,-fPIC,-fvisibility=hidden', '-Xlinker', '-rpath,/root/.theano/compiledir_Linux-3.19--generic-x86_64-with-Ubuntu-14.04-trusty-x86_64-2.7.6-64/cuda_ndarray', '-I/usr/local/lib/python2.7/dist-packages/theano/sandbox/cuda', '-I/usr/local/cuda/include', '-I/usr/local/lib/python2.7/dist-packages/numpy/core/include', '-I/usr/include/python2.7', '-I/usr/local/lib/python2.7/dist-packages/theano/gof', '-o', '/root/.theano/compiledir_Linux-3.19--generic-x86_64-with-Ubuntu-14.04-trusty-x86_64-2.7.6-64/tmpOSxzD1/bd1168e6726f16baed8c5f60c7ded9d1.so', 'mod.cu', '-L/usr/lib', '-lcudnn', '-lpython2.7', '-lcudart']

 

... and 

 

Exception: ('The following error happened while compiling the node', GpuDnnConvDesc{border_mode=(1, 1), subsample=(1, 1), conv_mode='conv', precision='float32'}(MakeVector{dtype='int64'}.0, MakeVector{dtype='int64'}.0), '\n', 'nvcc return status', 2, 'for cmd', 'nvcc -shared -O3 -arch=sm_61 -m64 -Xcompiler -fno-math-errno,-Wno-unused-label,-Wno-unused-variable,-Wno-write-strings,--machine=64,-DCUDA_NDARRAY_CUH=c72d035fdf91890f3b36710688069b2e,-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION,-fPIC,-fvisibility=hidden -Xlinker -rpath,/root/.theano/compiledir_Linux-3.19--generic-x86_64-with-Ubuntu-14.04-trusty-x86_64-2.7.6-64/cuda_ndarray -I/usr/local/lib/python2.7/dist-packages/theano/sandbox/cuda -I/usr/local/cuda/include -I/usr/local/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/theano/gof -o /root/.theano/compiledir_Linux-3.19--generic-x86_64-with-Ubuntu-14.04-trusty-x86_64-2.7.6-64/tmpOSxzD1/bd1168e6726f16baed8c5f60c7ded9d1.so mod.cu -L/usr/lib -lcudnn -lpython2.7 -lcudart', "[GpuDnnConvDesc{border_mode=(1, 1), subsample=(1, 1), conv_mode='conv', precision='float32'}(<TensorType(int64, vector)>, <TensorType(int64, vector)>)]")

 

 

the version of theano is not matched with the code. update the theano will be ok. 

 

11. Theano framework shown an error like this: 

　　File "/usr/local/lib/python2.7/dist-packages/enum.py", line 199, in __init__     raise EnumBadKeyError(key) 

　　Need 2 paramers (given 4)  

　　　　

　　==>> This error is caused when the enum34 module has been installed alongside the old enum module. enum34 is the backport for Python 2.x of the standard enum in Python 3.4. Many packages have started to use it and so it will be installed implicitly while installing another package. enum34 overrides the old enum files and causes this error.

You could remove enum34 and get rid of this error. But since Python 3.x has already adapted a new enum type, it might be wiser to uninstall the old enum and rewrite your code to use enum34. Its syntax is shown in this example.   This explnation comes from: https://codeyarns.com/2015/07/16/attributeerror-with-python-enum/ 

　　==>> maybe first you need to do is just install: $ sudo pip install enum34  

　　==>> The second main reason is that: you are using pkle file generted  not by your current environments, it's also caused by enum, this is a bug !

　　==>> So, you can re-generate your pkle file and try again. It will be OK. 

12. Load pkl file in python:

　　## Download the mnist dataset if it is not yet available. 

　　url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz' 

　　filename = 'mnist.pkl.gz' 

　　if not os.path.exists(filename):

　　　　print("Downloading mnist dataset ... ") 

　　　　urlretrieve(url, filename) 

　　## we will then load and unpickle the file. 

　　import gzip 

　　with gzip.open(filename, 'rb') as f:

　　　　data = pickle_load(f, encoding='latin-1') 

 

13. ValueError: You are tring to use the old GPU back-end. It was removed from Theano. Use device=cuda* now. 

 

wangxiao@AHU-Wangxiao:/media/wangxiao/724eaeef-e688-4b09-9cc9-dfaca44079b2/saliency-salgan-2017-master-tracking/scripts--suntao--finalVersion$ python wangxiao-02-train.py
Traceback (most recent call last):
File "wangxiao-02-train.py", line 11, in <module>
import theano
File "/usr/local/lib/python2.7/dist-packages/theano/__init__.py", line 67, in <module>
from theano.configdefaults import config
File "/usr/local/lib/python2.7/dist-packages/theano/configdefaults.py", line 119, in <module>
in_c_key=False)
File "/usr/local/lib/python2.7/dist-packages/theano/configparser.py", line 285, in AddConfigVar
configparam.__get__(root, type(root), delete_key=True)
File "/usr/local/lib/python2.7/dist-packages/theano/configparser.py", line 333, in __get__
self.__set__(cls, val_str)
File "/usr/local/lib/python2.7/dist-packages/theano/configparser.py", line 344, in __set__
self.val = self.filter(val)
File "/usr/local/lib/python2.7/dist-packages/theano/configdefaults.py", line 98, in filter
'You are tring to use the old GPU back-end. '
ValueError: You are tring to use the old GPU back-end. It was removed from Theano. Use device=cuda* now. See https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29 for more information.

==>>  just replace your gpu back-end with the suggestions, i.e. device=cuda* 　　

　　  vim ~/.theanorc  

　　and add these lines into this file: 

　　[global]

　　device = cuda

　　floatX=float32

　　[nvcc]

　　flags=--machine=64

　　[lib] cnmem=100 　　

 

 

 

14. Can not use cuDNN on context None: cannot compile with cuDNN. We got this error:

wangxiao@wangxiao:/media/wangxiao/724eaeef-e688-4b09-9cc9-dfaca44079b2/saliency-salgan-2017-master-tracking/scripts--suntao--finalVersion$ python wangxiao-02-train.py
Can not use cuDNN on context None: cannot compile with cuDNN. We got this error:
/usr/bin/ld: cannot find -lcudnn
collect2: error: ld returned 1 exit status 

==>> vim .theanorc 

[global]
device = cuda
floatX = float32

[dnn]
enabled = True
include_path=/usr/local/cuda/include
library_path=/usr/local/cuda/lib64

 

==>> This will make the cuDNN work. 

Using cuDNN version 5110 on context None
Mapped name None to device cuda: GeForce GTX 1080 (0000:02:00.0) 

 

15. theano && Lasagne, ImportError: No module named cuda 

 

Traceback (most recent call last):
File "wangxiao-02-train.py", line 17, in <module>
from models.model_salgan import ModelSALGAN
File "/media/wangxiao/724eaeef-e688-4b09-9cc9-dfaca44079b2/saliency-salgan-2017-master-tracking/scripts--suntao--finalVersion/models/model_salgan.py", line 7, in <module>
import generator
File "/media/wangxiao/724eaeef-e688-4b09-9cc9-dfaca44079b2/saliency-salgan-2017-master-tracking/scripts--suntao--finalVersion/models/generator.py", line 11, in <module>
import C3D_AlexNet
File "/media/wangxiao/724eaeef-e688-4b09-9cc9-dfaca44079b2/saliency-salgan-2017-master-tracking/scripts--suntao--finalVersion/models/C3D_AlexNet.py", line 14, in <module>
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
File "/usr/local/lib/python2.7/dist-packages/lasagne/layers/dnn.py", line 2, in <module>
from theano.sandbox.cuda import dnn
ImportError: No module named cuda 

 

==>>  change the following codes: 

export LD_LIBRARY_PATH=/home/wangxiao/cuda-v5.1/lib64:$LD_LIBRARY_PATH
export CPATH=/home/wangxiao/cuda-v5.1/include:$CPATH
export LIBRARY_PATH=/home/wangxiao/cuda-v5.1/lib64:$LIBRARY_PATH 


16. Error: pygpu.gpuarray.GpuArrayException: out of memory

0%| | 0/301 [00:00<?, ?it/s]Traceback (most recent call last):
File "wangxiao-02-train.py", line 231, in <module>
train()
File "wangxiao-02-train.py", line 217, in train
salgan_batch_iterator(model, train_data, None)
File "wangxiao-02-train.py", line 148, in salgan_batch_iterator
G_obj, D_obj, G_cost = model.G_trainFunction(batch_input,batch_output,batch_patch)
File "/home/wangxiao/anaconda2/lib/python2.7/site-packages/theano/compile/function_modul
