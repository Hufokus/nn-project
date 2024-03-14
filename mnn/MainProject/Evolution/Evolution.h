#ifndef _ENVIRONMENT_
#define _ENVIRONMENT_

#include "../Engine/Environment/environment.h"
#include "../NN/Network.h"


#include <unordered_map>

namespace evolution
{

	class Creature
	{
	public:
		Creature(std::shared_ptr<engine::environment::Player>&& p)
			: player(std::move(p))
		{
			//player->angularSpeed = 0.10000005f;

			//player->angularSpeed = 2.0f;
			//player->worldAngle = -1.3f;//3.14f;
			player->envCharge = 0.01f;
			//player->speed = { 0.0f, 10.0f };
		}

		void ProcessData();
		
		void SetNN(std::unique_ptr<Network> nn)
		{
			this->nn = std::move( nn);
		}

	private:
		std::shared_ptr<engine::environment::Player> player;
		std::unique_ptr<Network> nn;
		std::vector<float> output;


		friend class EvolutionControl;
	};

	class EvolutionControl
	{
	public:
		EvolutionControl()
		{}

		void Init();
		void ProcessCreature();
		void SwitchToNextGeneration();
		void ProcessEvolution();
		

		void SetScene(engine::environment::Scene* scene)
		{
			this->scene = scene;
		}

		void onCreatureDies();

	private:

		engine::environment::Scene* scene;
		std::unique_ptr<Creature> creature;

		std::unique_ptr<Network> baseGenNN;
		std::vector<std::pair<Network, std::chrono::steady_clock::duration>> allGenNNs;

		uint16_t epochCount{ 0 };
		std::optional<uint16_t> generationCount{};
		uint16_t specimenCount{ 0 };

		std::chrono::steady_clock::time_point creatureLifeStart;
		std::chrono::steady_clock::duration creatureLifeDuration;

		bool isCreatureAlive{ true };

	};


}



#endif /*_ENVIRONMENT_H*/
