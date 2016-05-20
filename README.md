Caffe Model Compression
===================


This is a python tool used to compress the trained caffe weights.  For Alexnet,  we got 17x compression rate (~233M bytes to 14M bytes).  The idea comes from [Deep Compression](http://arxiv.org/pdf/1510.00149v5.pdf) . This work does not implement purning and Huffman coding, but implement the Kmeans -based quantization to compress the weights of convolution and full-connected layer.  One contribution of this work is using OpenMP to accelerate the Kmeans processing.

----------


####Dependency

> - Python/Numpy
> - Caffe


####Authors
> - [Li, Victor](yuanyuan.li85@gmail.com)
> - [Qiu, Junyi](geoffrey0824@sjtu.edu.cn)

####How to Build:
```
cd quantz_kit 
 ./build.sh
```
####How to use it:
```
caffe_model_compress: #function to compress model 
caffe_model_decompress: #function to decompress model 
```

