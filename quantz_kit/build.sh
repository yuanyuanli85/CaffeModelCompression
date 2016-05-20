#!/bin/sh

#swig -python weights_compress.i
g++ -O3 -mavx2 -Wno-cpp -std=c++11 -fopenmp -fPIC -c weights_compress.cpp weights_compress_wrap.c -I/usr/local/include/ -I/usr/include/python2.7/ -I/usr/include/numpy -I.
#g++ -Wno-cpp -std=c++11 -fopenmp -fPIC -c KmDriver.cpp -I/usr/local/include/ -I/usr/include/python2.7/ -I/usr/include/numpy -I.
#g++ -Wno-cpp -std=c++11 -fopenmp -fPIC -c KmPointer.cpp -I/usr/local/include/ -I/usr/include/python2.7/ -I/usr/include/numpy -I.
g++ -fopenmp -shared weights_compress.o weights_compress_wrap.o -o _weights_quantization.so
