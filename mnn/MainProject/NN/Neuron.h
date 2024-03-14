#ifndef _NEURON_H
#define _NEURON_H



#include <memory>

class Layer;

class Neuron
{




private:


	float* _WeightsAddr;

	std::unique_ptr<float> _Distances;

	std::shared_ptr<Layer> _MyLayer;

};




#endif
