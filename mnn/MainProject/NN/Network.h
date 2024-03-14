#ifndef _NETWORK_H
#define _NETWORK_H

#include "Layer.h"

#include <memory>
#include <stdint.h>
#include <optional>
#include <cassert>

class Network
{
public:

	Network()
	{

	}

	Network(const Network& other)
	{

		std::vector<std::pair<std::shared_ptr<BaseLayer> /*old pointer*/, std::shared_ptr<BaseLayer> /*new pointer*/>> transferLayersData;

		for( const auto& l : other._InputLayers )
			transferLayersData.emplace_back( std::make_pair( l, _InputLayers.emplace_back( std::make_shared<InputLayer>( *l))));

		for( const auto& l : other._NeuralLayers )
			transferLayersData.emplace_back( std::make_pair( l, _NeuralLayers.emplace_back( std::make_shared<NeuralLayer>( *l))));


		for( auto& l : _NeuralLayers )
		{
			for( auto& fl : l->_FeedingLayers )
			{
				const auto& it = std::find_if(transferLayersData.begin(), transferLayersData.end(), [&](std::pair<std::shared_ptr<BaseLayer>, std::shared_ptr<BaseLayer>>& d) {
					return fl.first == d.first;
				});

				if( it != transferLayersData.end() )
					fl.first = it->second;
				else 
					assert( false);

				fl.first->setNextLayer(l);
			}
			
		}

		

	}

	void AddInputLayer(uint16_t size);

	void AddNeuralLayer(uint16_t size, 
		std::vector<std::tuple<uint16_t, std::optional<float>, std::optional<float>>> inputLayerFeeders,
		std::vector<std::tuple<uint16_t, std::optional<float>, std::optional<float>>> neuralLayerFeeders);

	void PopulateWeightMatrices();
	void MutateWeights(float mutationRange);

	std::vector<float> GetOutput();

	std::shared_ptr<NeuralLayer> GetNeuralLayer(uint16_t pos);

	void AddNeuron(uint16_t layerPos, uint16_t pos);

	void RunNetwork();

	void SetInputData( uint16_t layer, std::vector<float> data);

	//void InputData( std::vector<std::vector<float>> inputs);

private:


	std::vector<std::shared_ptr<NeuralLayer>> _NeuralLayers;
	std::vector<std::shared_ptr<InputLayer>> _InputLayers;



};

#endif
