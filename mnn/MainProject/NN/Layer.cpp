#include "Layer.h"
#include "Network.h"
#include "matrix_supp.h"

#include "CudaCalcs/cuda_calc.h"

#include <algorithm>
#include <cassert>


//#define LOG_NEURAL_LAYER_OUTPUT 


constexpr float MATRIX_RESERVE_MUL = 2.0f;
constexpr float NEURONS_RESERVE_MUL = 1.5f;


NeuralLayer::NeuralLayer(uint16_t neuronsCount)
	: BaseLayer( neuronsCount)
{
}


void NeuralLayer::InsertRowOfWeights(uint16_t pos, std::shared_ptr<BaseLayer> layer)
{
	//auto it = _FeedingLayers.find( layer);
	auto it = std::find_if(_FeedingLayers.begin(), _FeedingLayers.end(), [&](const auto& obj) {
		return obj.first == layer;
	});

	if( it != _FeedingLayers.end() )
	{
		mx::InsertRow( it->second._weights, pos, 0.01f, NeuronsCount());
	}
}

void NeuralLayer::ConnectToFeeders(std::weak_ptr<NeuralLayer> weak_this, 
	std::vector<std::tuple<std::shared_ptr<BaseLayer>, std::optional<float>, std::optional<float>>> feedingLayers)
{
	for( auto& fls : feedingLayers )
	{
		auto [ fl, averageWeight, distance ] = std::move( fls);

		uint16_t rows = fl->NeuronsCount();
		auto elemsCount = NeuronsCount() * rows;

		

		std::vector<float> w, f, id, cd, g;
		w.reserve( MATRIX_RESERVE_MUL * elemsCount );

		if( averageWeight.has_value() )
		{
			
			w.assign(elemsCount, averageWeight.value());
			for( auto& ww : w )
			{
				float low = -1.0f;
				float high = 1.0f;
				float r = low + static_cast<float>( rand() ) * static_cast<float>( high - low ) / RAND_MAX;
				ww = r; /*static_cast <float> (rand()) / static_cast <float> ( RAND_MAX );*/
			}

		} else {

			// fill with random values, but for now fill with zeroes
			w.assign(elemsCount, 0.0f);
		}

		f.reserve(MATRIX_RESERVE_MUL * elemsCount);

		id.reserve(MATRIX_RESERVE_MUL * elemsCount);

		g.reserve(MATRIX_RESERVE_MUL * elemsCount);

		cd.reserve(MATRIX_RESERVE_MUL * elemsCount);


		/*
			More of the matrices initialization required.
		
		*/



		fl->setNextLayer(weak_this);

		_FeedingLayers.emplace_back(std::move(fl), ParamMatrices{ std::move(w), std::move(f), std::move(g), std::move(id), std::move(cd) });

	}

}

void NeuralLayer::AddNeuron(uint16_t pos, uint16_t count)
{
	BaseLayer::AddNeuron( pos, count);

	for( auto& wtf : _FeedingLayers )
	{
		mx::InsertColumn( wtf.second._weights, pos, 0.01f, NeuronsCount());
	}
}

void NeuralLayer::RunLayer()
{

	PrepareAndCalculate( this);

#ifdef LOG_NEURAL_LAYER_OUTPUT
	printf("\n\n============================================================\n");
	printf("============================================================");
	printf("\n============================================================");
#endif 

	for( const auto& fl : _FeedingLayers )
	{

#ifdef LOG_NEURAL_LAYER_OUTPUT
		printf("\n\n");
		printf( "\nFeeding layer output:\n");
		printf("\n");
		for( const auto& o : fl.first->_Output )
			printf("  %.3f", o);

		printf("\n\n");
		printf("\nWeights for this feeding layer:\n");
		printf("\n");
#endif 

		int r = 0;
		for( int i = 0; i < fl.second._weights.size(); i++ )
		{
			if( r >= _Output.size() )
			{

#ifdef LOG_NEURAL_LAYER_OUTPUT
				printf("\n");
#endif

				r = 0;
			}

#ifdef LOG_NEURAL_LAYER_OUTPUT
			printf( "  %.3f", fl.second._weights [i]);
#endif

			r++;

		}

#ifdef LOG_NEURAL_LAYER_OUTPUT
		printf("\n\n");
#endif

	}

#ifdef LOG_NEURAL_LAYER_OUTPUT
	printf("\nOutput: \n\n");
	for( const auto& o : _Output )
		printf("  %.3f", o);

	printf("\n");
#endif


	/*uint16_t feedingSizeTotal{ 0 };
	uint16_t weightsSizeTotal{ 0 };


	for( auto& fl : _FeedingLayers )
	{
		feedingSizeTotal += fl.first->_Output.size();
		weightsSizeTotal += fl.second._weights.size();
	}

	int feedingBytes = feedingSizeTotal * sizeof(float);
	int weightsBytes = weightsSizeTotal * sizeof(float);
	int outputBytes = _Output.size() * sizeof(float);

	float* d_feedingNeurons;
	float* d_weights;
	float* d_output;

	cudaMalloc( d_feedingNeurons, feedingBytes);
	cudaMalloc( d_weights, weightsBytes);
	cudaMalloc( d_output, outputBytes);

	uint16_t feedingPos{ 0 };
	uint16_t weightsPos{ 0 };

	for( auto& fl : _FeedingLayers )
	{
		int size = fl.first->_Output.size();
		cudaMemcpy( d_feedingNeurons [feedingPos], fl.first->_Output.data(), size * sizeof(float), cudaMemcpyHostToDevice);
		feedingPos += size;

		size = fl.second._weights.size();
		cudaMemcpy( d_weights [weightsPos], fl.second._weights.data(), size * sizeof(float), cudaMemcpyHostToDevice);
		weightsPos += size;
	}
	

	int threadsPerBlock = 32;
	int blocksPerGrid = (_Output.size() + threadsPerBlock - 1) / threadsPerBlock;

	VecMatMul<<<blocksPerGrid, threadsPerBlock>>>( d_feedingNeurons, d_weights, d_output, feedingSizeTotal, _Output.size());

	cudaDeviceSynchronize();

	cudaMemcpy( _Output.data(), d_output, outputBytes, cudaMemcpyDeviceToHost);

	cudaFree( d_feedingNeurons);
	cudaFree( d_weights);
	cudaFree( d_output);*/

}





void InputLayer::SetData(std::vector<float> data)
{
	setData( std::move( data));
}

void BaseLayer::setData(std::vector<float> data)
{
	if( data.size() != _Output.size() )
	{
		assert( false);
		return;
	}

	_Output = std::move(data);
}



BaseLayer::BaseLayer(uint16_t neuronsCount)
	: _NeuronsCount(neuronsCount)
{
	_Output.reserve(neuronsCount * NEURONS_RESERVE_MUL);
	_Output.assign(neuronsCount, 0.0f);
}

uint16_t BaseLayer::NeuronsCount() const
{
	return _NeuronsCount;
}

void BaseLayer::setNeuronsCount(uint16_t n)
{
	_NeuronsCount = n;
}

void BaseLayer::AddNeuron(uint16_t pos, uint16_t count)
{

	_Output.insert( _Output.begin() + pos, count, 0.0f);
	setNeuronsCount( _Output.size());

	for( auto& nl : _NextLayers )
	{
		if( auto pnl = nl.lock())
			pnl->InsertRowOfWeights(pos, shared_from_this());
	}

}

void BaseLayer::incNeuronCount(uint16_t count)
{
	_NeuronsCount += count;
}


void BaseLayer::setNextLayer(std::weak_ptr<NeuralLayer> nextLayer)
{
	_NextLayers.emplace_back(nextLayer);
}