#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <limits>
#include <sys/time.h>
#include <Python.h>
#include "numpy/arrayobject.h"
#include <omp.h>
//#include "KmDriver.h"


using namespace std;

void kmeans_cluster(int *cLabel, float *cCentro, float *cNodes, int nNode, int nDimension, int nCluster, float *cInitCentro=nullptr, int max_iter=1000)
{ 
	// get threads number from the env variable
	int tSize;
	if (const char* env_omp_tnum = getenv("OMP_NUM_THREADS"))
		tSize = atoi(env_omp_tnum);
	else
		tSize = omp_get_num_procs();
		
	// generate initial centroids
	float max_global=numeric_limits<float>::min(), min_global=numeric_limits<float>::max();
	float max_private[tSize], min_private[tSize];
	if (cInitCentro != nullptr)
		memcpy(cCentro, cInitCentro, sizeof(float)*nCluster*nDimension);
	else
	{
#pragma omp parallel
{
		float max=numeric_limits<float>::min(), min=numeric_limits<float>::max();
		#pragma omp for
		for (int n=0; n<nNode; n++)
		{
			max = (max < cNodes[n]) ? cNodes[n] : max;
			min = (min > cNodes[n]) ? cNodes[n] : min;
		}
		int tid = omp_get_thread_num();
		max_private[tid]=max;
		min_private[tid]=min;
}
		for (int i=0; i<tSize; i++)
		{
			max_global = (max_global < max_private[i]) ? max_private[i] : max_global; 
			min_global = (min_global > min_private[i]) ? min_private[i] : min_global; 
		}
		cInitCentro = new float[nCluster];
		for (int k=0; k<nCluster; k++)
			cCentro[k] = min_global + (max_global-min_global)*k/(nCluster-1);
	}

	const float float_max = numeric_limits<float>::max();

	// initialize
	float *cDistance = new float[nNode*nDimension];
	int *cClusterSize = new int[nCluster];

	float *pCentroPos = new float[nCluster*tSize];
	int *pClusterSize = new int[nCluster*tSize];
	memset(pClusterSize, 0, sizeof(int)*nCluster*tSize);
	memset(pCentroPos, 0, sizeof(float)*nCluster*tSize);

	int iter = 0;
	float tk1=0.f, tk2=0.f, tk3=0.f;
	double mCurDistance = 0.0;
	double mPreDistance = numeric_limits<double>::max();

	// clustering
    while (iter < max_iter)
    {
		// check convergence
	    if (fabs(mPreDistance-mCurDistance)/mPreDistance < 0.01) break;
	    mPreDistance = mCurDistance;
	    mCurDistance = 0.0;

		// select nearest cluster
#pragma omp parallel
{
        #pragma omp for //nowait
        for(int n=0; n<nNode; n++)
        {
            float distance;
            float mindistance = float_max;	
            int clostCluster;
            for(int k=0; k<nCluster; k++)
            {
                distance = fabs(cNodes[n] - cCentro[k]);
                if (distance < mindistance)
                {
                    mindistance = distance;
                    clostCluster = k;
                }
            }
            cDistance[n] = mindistance;
            cLabel[n] = clostCluster;
        }
}

		// calc new distance/inertia
#pragma omp parallel
{
        #pragma omp for reduction(+:mCurDistance) //nowait
        for(int n=0; n<nNode; n++)
            mCurDistance = mCurDistance + cDistance[n];
}

		// generate new centroids
		// accumulation(private)
#pragma omp parallel
{
        float ptrC[nCluster];
        int ptrS[nCluster];
        for(int k=0; k<nCluster; k++)
        {
            ptrC[k] = 0.f;
            ptrS[k] = 0;
        }

        #pragma omp for nowait
        for(int n=0; n<nNode; n++)
        {
            ptrC[cLabel[n]] += cNodes[n]; 
            ptrS[cLabel[n]] += 1;
        }

        int tid = omp_get_thread_num();
        for(int k=0; k<nCluster; k++)
        {
            pCentroPos[tid*nCluster+k] = ptrC[k];
            pClusterSize[tid*nCluster+k] = ptrS[k];
        }
}
		//reduction(global)
        for(int k=0; k<nCluster; k++)
        {
            cCentro[k] = 0.f;
            cClusterSize[k] = 0;
            for(int i=0; i<tSize; i++)
            {
                cCentro[k] += pCentroPos[i*nCluster+k];
                cClusterSize[k] += pClusterSize[i*nCluster+k];
            }
            cCentro[k] /= cClusterSize[k];
        }

        iter++;
//	cout << "Iteration: " << iter << " Distance: " << mCurDistance << endl;
    }
    //gather centroids
    //#pragma omp parallel for
    //for(int n=0; n<nNode; n++)
    //    cNodes[n] = cCentro[cLabel[n]];

    delete [] cDistance;
    delete [] cClusterSize;
    delete [] pClusterSize;
    delete [] pCentroPos;

}

void encode_label(int *cmprLabel, int *orgLabel, int num, const int nbit)
{
	int mask = 1;
	for(int i=1; i<nbit; i++)
		mask |= (mask << 1);
	
	const int ngroup = sizeof(int)*8 / nbit;

	// proceed the main loop
	#pragma omp parallel for
	for (int nt=0; nt<num/ngroup; nt++)
	{
		cmprLabel[nt] &= 0x00000000;
		int *cGroup = orgLabel+nt*ngroup;
		int *cToken = cmprLabel+nt;
		for (int ng=0; ng<ngroup; ng++)
			*cToken |= cGroup[ng] << ((ngroup-1-ng)*nbit);
	}

	{ // proceed the rest
	int *cGroup = orgLabel+(num/ngroup)*ngroup;
	int *cToken = cmprLabel+num/ngroup;
	*cToken &= 0x00000000;
	for (int ng=0; ng<(num%ngroup); ng++)
		*cToken |= cGroup[ng] << ((ngroup-1-ng)*nbit);
	}


/*	else
	{
		// nbit = 5, 6, 7, ... , etc.
		int lcm = nbit;
		while (lcm % 2 == 0) lcm /= 2;
		const int ntoken = lcm; 
		lcm = lcm*sizeof(int)*8;
		const int ngroup = lcm / nbit;

		// proceed the main loop
		#pragma omp parallel for
		for (int ng=0; ng<num/ngroup; ng++)
		{
			int nres = 0;
			int __uname = 0;
			for(int nt=0; nt<ntoken; nt++)
			{
				cmprLabel[nt] = 0x00000000;
				int nelem = (sizeof(int)*8-nres) / nbit;
				if (nres != 0)
					cmprLabel[nt] |= orgLabel[ng*ngroup+__uname-1] << (32-nres);
				for(int ne=0; ne<nelem; ne++)
				{
					cmprLabel[nt] |= orgLabel[ng*ngroup+__uname+ne] << (32-nres-(ne+1)*nbit);
				}
				nres = ((nelem+1)*nbit+nres-32) % nbit;
				if (nres != 0)
					cmprLabel[nt] |= orgLabel[ng*ngroup+__uname+nelem] >> nres;
				__uname += nelem;
			}
		}

		// proceed the rest
		if (num % ngroup != 0)
		{
			int ng = num/ngroup;
			int nres = 0;
			int __uname = 0;
			for(int nt=0; nt<ntoken; nt++)
			{
				cmprLabel[nt] = 0x00000000;
				int nelem = (sizeof(int)*8-nres) / nbit;
				if (nres != 0)
					cmprLabel[nt] |= orgLabel[ng*ngroup+__uname-1] << (32-nres);
				for(int ne=0; ne<nelem; ne++)
				{
					if (ng*ngroup+__uname+ne >= num) break;
					cmprLabel[nt] |= orgLabel[ng*ngroup+__uname+ne] << (32-nres-(ne+1)*nbit);
				}
				if ((ng*ngroup+__uname+nelem) >= num) break;
				nres = (nelem+1)*nbit+nres-32;
				if (nres != 0)
					cmprLabel[nt] |= orgLabel[ng*ngroup+__uname+nelem] >> nres;
				__uname += nelem + 1;
			}
		}

	}*/
}

void decode_label(int *dcmprLabel, int *cmprLabel, int num, const int nbit)
{
	int mask = 1;
	for(int i=1; i<nbit; i++)
		mask |= (mask << 1);
	
	const int ngroup = sizeof(int)*8 / nbit;

	#pragma omp parallel for
	for (int nt=0; nt<num/ngroup; nt++)
	{
		int *cGroup = dcmprLabel+nt*ngroup;
		int token = cmprLabel[nt];
		for (int ng=0; ng<ngroup; ng++)
			cGroup[ng] = (token >> ((ngroup-1-ng)*nbit)) & mask;
	}
	{
	int *cGroup = dcmprLabel+(num/ngroup)*ngroup;
	int token = *(cmprLabel+num/ngroup);
	for (int ng=0; ng<(num%ngroup); ng++)
		cGroup[ng] = (token >> ((ngroup-1-ng)*nbit)) & mask;
	}
}

void compress_layer_weights(PyObject *pComprsLabel, PyObject *pCodeBook, PyObject *pWeights, int nWeight, int nBit)
{
	float *cWeights = (float *) PyArray_GETPTR1(pWeights, 0);
	float *cCodeBook = (float *) PyArray_GETPTR1(pCodeBook, 0);
	int *cLabel = new int[nWeight];
	int nCentroid = 1 << nBit;
	float *cCentroid = new float[nCentroid];
	int *cCmprLabel = (int *) PyArray_GETPTR1(pComprsLabel, 0);

	kmeans_cluster(cLabel, cCentroid, cWeights, nWeight, 1, nCentroid, nullptr, 1000);
	memcpy(cCodeBook, cCentroid, nCentroid*sizeof(float));
	encode_label(cCmprLabel, cLabel, nWeight, nBit);
}

void decompress_layer_weights(PyObject *pWeights, PyObject *pComprsLabel, PyObject *pCodeBook, int nWeight, int nBit)
{
	int nCentroid = 1 << nBit;
	int *cLabel = new int[nWeight];
	int *cCmprLabel = (int *) PyArray_GETPTR1(pComprsLabel, 0);
	float *cWeights = (float *) PyArray_GETPTR1(pWeights, 0);
	float *cCodeBook = (float *) PyArray_GETPTR1(pCodeBook, 0);

	decode_label(cLabel, cCmprLabel, nWeight, nBit);
	//translate
	#pragma omp parallel for
	for(int n=0; n<nWeight; n++)
		cWeights[n] = cCodeBook[cLabel[n]];
}

void quantize_layer_weights(PyObject *pWeights, int nWeight, int nBit)
{
	float *cWeights = (float *) PyArray_GETPTR1(pWeights, 0);
	int *cLabel = new int[nWeight];
	int nCentroid = 1 << nBit;
	float *cCentroid = new float[nCentroid];

	kmeans_cluster(cLabel, cCentroid, cWeights, nWeight, 1, nCentroid, nullptr, 1000);
	//translate
	#pragma omp parallel for
	for(int n=0; n<nWeight; n++)
		cWeights[n] = cCentroid[cLabel[n]];
}
