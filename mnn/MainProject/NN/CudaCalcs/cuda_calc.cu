
#include "../Layer.h"

#include "cuda_calc.h"
#include <stdio.h>




__global__ void mm()
{
	printf("====================== test ================\n");
};

__global__ void VecMatMul( float* v, float* m, float* out, int v_size, int m_columns)
{
	float c = 0.0f;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if( col > m_columns)
		return;

	for( int i = 0; i < v_size; ++i )
	{
		c += v [i] * m [i * m_columns + col];
	}
	out [col] = tanh( c);
};


void PrepareAndCalculate(NeuralLayer* layer)
{
	uint16_t feedingSizeTotal{ 0 };
	uint16_t weightsSizeTotal{ 0 };
	

	for( auto& fl : layer->_FeedingLayers )
	{
		feedingSizeTotal += fl.first->_Output.size();
		weightsSizeTotal += fl.second._weights.size();
	}

	int feedingBytes = feedingSizeTotal * sizeof(float);
	int weightsBytes = weightsSizeTotal * sizeof(float);
	int outputBytes = layer->_Output.size() * sizeof(float);

	float* d_feedingNeurons;
	float* d_weights;
	float* d_output;

	cudaMalloc(&d_feedingNeurons, feedingBytes);
	cudaMalloc(&d_weights, weightsBytes);
	cudaMalloc(&d_output, outputBytes);

	uint16_t feedingPos{ 0 };
	uint16_t weightsPos{ 0 };

	for( auto& fl : layer->_FeedingLayers )
	{
		int size = fl.first->_Output.size();
		cudaMemcpy(&d_feedingNeurons [feedingPos], fl.first->_Output.data(), size * sizeof(float), cudaMemcpyHostToDevice);
		feedingPos += size;

		size = fl.second._weights.size();
		cudaMemcpy(&d_weights [weightsPos], fl.second._weights.data(), size * sizeof(float), cudaMemcpyHostToDevice);
		weightsPos += size;
	}


	int threadsPerBlock = 32;
	int blocksPerGrid = ( layer->_Output.size() + threadsPerBlock - 1 ) / threadsPerBlock;

	VecMatMul << <blocksPerGrid, threadsPerBlock >> > ( d_feedingNeurons, d_weights, d_output, feedingSizeTotal, layer->_Output.size() );

	cudaDeviceSynchronize();

	cudaMemcpy(layer->_Output.data(), d_output, outputBytes, cudaMemcpyDeviceToHost);

	cudaFree(d_feedingNeurons);
	cudaFree(d_weights);
	cudaFree(d_output);

}

void MatrixMul()
{

	mm<<<1,4>>>();
	cudaDeviceSynchronize();
};

