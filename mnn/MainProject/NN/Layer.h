#ifndef _LAYER_H
#define _LAYER_H

#include <vector>
#include <memory>
#include <functional>
#include <stdint.h>
#include <map>
#include <optional>

class NeuralLayer;
class Network;

class ParamMatrices
{
public:

	ParamMatrices(std::vector<float> w, std::vector<float> f, std::vector<float> g, std::vector<float> id, std::vector<float> cd)
		: _weights(std::move(w)), _fadingFactors(std::move(f)),
		_growingFactors(std::move(g)),
		_initialDistances(std::move( id)), _currentDistances( std::move( cd))
	{
	}

	std::vector<float> _weights;
	std::vector<float> _fadingFactors;
	std::vector<float> _growingFactors;
	std::vector<float> _initialDistances;
	std::vector<float> _currentDistances;

};


class BaseLayer : public std::enable_shared_from_this<BaseLayer>
{
public:

	BaseLayer(uint16_t neuronsCount);

	BaseLayer(const BaseLayer& other) :
		_Output( other._Output),
		_NeuronsCount( other._NeuronsCount)
	{

	}

	uint16_t NeuronsCount() const;
	virtual void AddNeuron(uint16_t pos, uint16_t count = 1);

protected:

	void setData(std::vector<float> data);
	void setNeuronsCount( uint16_t n);
	//std::vector<std::shared_ptr<NeuralLayer>>& nextLayers();

	void incNeuronCount( uint16_t count = 1);

private:


	void setNextLayer(std::weak_ptr<NeuralLayer> nextLayer);

	uint16_t _NeuronsCount{ 0 };
	std::vector<std::weak_ptr<NeuralLayer>> _NextLayers;
	std::vector<float> _Output;

	friend NeuralLayer;
	friend Network;
	friend void PrepareAndCalculate(NeuralLayer* layer);
};


class InputLayer : public BaseLayer
{
public:
	InputLayer(uint16_t neuronsCount)
		: BaseLayer( neuronsCount)
	{
	}

	void SetData(std::vector<float> data);

private:



};


class NeuralLayer : public BaseLayer
{
public:

	NeuralLayer(uint16_t neuronsCount);

	NeuralLayer(const NeuralLayer& other) // to be used from NN only
		: BaseLayer(other)
	{
		_FeedingLayers = other._FeedingLayers;  // shared_ptrs must be replaced with new values
	}


	void InsertRowOfWeights( uint16_t pos, std::shared_ptr<BaseLayer> layerToConnect);
	void ConnectToFeeders(std::weak_ptr<NeuralLayer> weak_this, std::vector<std::tuple<std::shared_ptr<BaseLayer>, std::optional<float>, std::optional<float>>> feedingLayers);
	virtual void AddNeuron(uint16_t pos, uint16_t count = 1) override;
	void RunLayer();
	std::vector<float> GetOutput()
	{
		return _Output;
	}

	std::function<float(float)> _Activation {};



	
	std::vector<std::pair<std::shared_ptr<BaseLayer>, ParamMatrices>> _FeedingLayers;
private:
	friend void PrepareAndCalculate(NeuralLayer* layer);
	friend Network;
};





#endif
