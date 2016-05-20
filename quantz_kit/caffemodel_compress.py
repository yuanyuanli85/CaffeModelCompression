import os 
import caffe 
import numpy as np
from math import ceil
#from quantz_kit.weights_quantization import weights_quantization
import sys
sys.path.append("./quantz_kit")
import weights_quantization as wqtz
import time


def caffe_model_compress(model, weights, storefile, convbit=6, fcbit=2, use_savez_compressed=True):
	net = caffe.Net(model, caffe.TEST);
	net.copy_from(weights);

	xdict = dict()
	#version 1 ; bits of conv layer and bits of full-connected layer
	xdict['compz_info'] = (1, int(convbit), int(fcbit))
	
	for item in net.params.items():
		name, layer = item
		print "compressing layer", name
		
		#compress weights
		weights = net.params[name][0].data
		#don't compress bais
		bais    = net.params[name][1].data
		
		#bits for conv and full-connected layer.
		if "fc" in name:
			nbit = int(fcbit)
		elif "conv" in name:
			nbit = int(convbit)

		weights_vec = weights.flatten().astype(np.float32)
		vec_length = weights_vec.size
		nelem = 32 / nbit
		newlabel = np.empty(((vec_length+nelem-1)/nelem),dtype=np.int32) 
		codebook = np.empty((2**nbit),dtype=np.float32)

		#t_start = time.time()
		wqtz.compress_layer_weights(newlabel, codebook, weights_vec, vec_length, nbit)
		#t_stop = time.time()
		#kmeans_time = kmeans_time + t_stop - t_start
				
		xdict[name+'_weight_labels'] = newlabel
		xdict[name+'_weight_codebook'] = codebook
		xdict[name+'_bias'] = bais

	#keep result into output file
	if (use_savez_compressed):
		np.savez_compressed(storefile, **xdict)
	else:
		np.savez(storefile, **xdict)

	
def caffe_model_decompress(model, weights, loadfile):
	net = caffe.Net(model, caffe.TEST);
	cmpr_model = np.load(loadfile)
	
	print cmpr_model.files
	
	version, convbit, fcbit = cmpr_model['compz_info']
	
	assert(version == 1), "compz version not support"
	
	
	for item in net.params.items():
		name, layer = item
		newlabels = cmpr_model[name+'_weight_labels']
		codebook = cmpr_model[name+'_weight_codebook']

		origin_size = net.params[name][0].data.flatten().size
		weights_vec = np.empty(origin_size, dtype=np.float32)
		vec_length = weights_vec.size
		
		#need have a way to get bits for fc and conv
		if "fc" in name:
			nbit = fcbit
		elif "conv" in name:
			nbit = convbit

		wqtz.decompress_layer_weights(weights_vec, newlabels, codebook, vec_length, nbit)
		newweights = weights_vec.reshape(net.params[name][0].data.shape)
		net.params[name][0].data[...] = newweights

		newbias = cmpr_model[name+'_bias']
		net.params[name][1].data[...] = newbias[...]
	net.save(weights)

if __name__ == "__main__":
	ALEXNET_PATH = "/home/junyi/caffe/models/bvlc_alexnet"
	netmodel   = os.path.join(ALEXNET_PATH, "deploy.prototxt") 
	netweights = os.path.join(ALEXNET_PATH, "bvlc_alexnet.caffemodel")
	output = "alexnetzip"
	caffe_model_compress(netmodel, netweights, output, 6, 2)

	caffe_model_decompress(netmodel, "alexnet_xx.caffemodel", "alexnetzip.npz")
	
	print "Done"
