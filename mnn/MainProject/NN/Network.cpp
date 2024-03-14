

#include "Network.h"
#include "Layer.h"
#include <cassert>

void Network::AddInputLayer(uint16_t size)
{
	_InputLayers.emplace_back( std::make_shared<InputLayer>( size));
}



void Network::AddNeuralLayer(uint16_t size, 
	std::vector<std::tuple<uint16_t, std::optional<float>, std::optional<float>>> inputLayerFeeders,
	std::vector<std::tuple<uint16_t, std::optional<float>, std::optional<float>>> neuralLayerFeeders)
{

	std::vector<std::tuple<std::shared_ptr<BaseLayer>, std::optional<float>, std::optional<float>>> layers;
	for( auto il : inputLayerFeeders )
	{
		auto [ n, w, d] = std::move( il);
		layers.emplace_back(_InputLayers [n], std::move( w), std::move( d));
	}

	for( auto nl : neuralLayerFeeders )
	{
		auto [n, w, d] = std::move(nl);
		layers.emplace_back( _NeuralLayers [n], std::move(w), std::move(d));
	}
	
	std::shared_ptr<NeuralLayer> nl = std::make_shared<NeuralLayer>( size);
	nl->ConnectToFeeders( nl, std::move( layers));
	_NeuralLayers.emplace_back( std::move( nl));
	
}

void Network::PopulateWeightMatrices()
{
	for( auto& nl : _NeuralLayers )
	{
		for( auto& fl : nl->_FeedingLayers )
		{
			for( auto& w : fl.second._weights )
			{
				float low = -1.0f;
				float high = 1.0f;
				float r = low + static_cast<float>( rand() ) * static_cast<float>( high - low ) / RAND_MAX;
				w = r;
			}
		}
	}
}

void Network::MutateWeights(float mutationRange)
{

	for( auto& nl : _NeuralLayers )
	{
		for( auto& fl : nl->_FeedingLayers )
		{
			for( auto& w : fl.second._weights )
			{
				float low = w - mutationRange; 
				float high = w + mutationRange; 
				if( low > high)
					std::swap( low, high);

				if( high > 1.0f )
						high = 1.0f;
				if( low < -1.0f)
						low = 1.0f;

				float r = low + static_cast<float>( rand() ) * static_cast<float>( high - low ) / RAND_MAX;
				w = r;
			}
				
		}
	}
}

std::vector<float> Network::GetOutput()
{
	return _NeuralLayers.back()->GetOutput();
}

std::shared_ptr<NeuralLayer> Network::GetNeuralLayer(uint16_t pos)
{
	if( pos >= _NeuralLayers.size() )
	{
		assert(false);
		return {};
	}

	return _NeuralLayers [pos];
}

void Network::AddNeuron(uint16_t layerPos, uint16_t pos)
{

	if( layerPos >= _NeuralLayers.size() )
	{
		assert(false);
		return;
	}

	_NeuralLayers [layerPos]->AddNeuron( pos);
}
void Network::RunNetwork()
{
	for( auto& nl : _NeuralLayers )
	{
		nl->RunLayer();
	}
}
void Network::SetInputData(uint16_t layer, std::vector<float> data)
{
	if( layer >= _InputLayers.size() )
	{
		assert(false);
		return;
	}

	_InputLayers [layer]->SetData( std::move( data));
}
//void Network::InputData(std::vector<std::vector<float>> inputs)
//{
//	for( int i = 0; i < inputs.size(); i++ )
//	{
//		if( _InputLayers [i].NeuronsCount() != inputs [i].size() )
//		{
//			assert(false);
//			return;
//		}
//
//		_InputLayers [i].SetData( std::move( inputs [i]));
//	}
//}
