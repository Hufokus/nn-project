
#include "Evolution.h"
#include <algorithm>

uint16_t GEN_SIZE = 200;

namespace evolution
{
	void evolution::EvolutionControl::Init()
	{

		scene->AddBorders(1200, 900, { 600, 500 });
		scene->AddFood(70, 70, { 200, 200 });
		//scene->AddFood(20, 20, { 600, 700 });

		scene->AddObstacle(40, 40, {800, 100});
		scene->AddObstacle(40, 40, { 450, 500 });
		scene->AddObstacle(40, 40, { 530, 200 });
		scene->AddThreat(15, 15, { 400, 400 });
		creature = std::make_unique<Creature>( scene->AddPlayer(50, 50, {600, 600}));	
		std::unique_ptr<Network> nn = std::make_unique<Network>();

		/// <summary>
		///  CHANGE WEIGHTS TO  0.0 ? 
		///  
		/// </summary>
		nn->AddInputLayer(40);
		nn->AddInputLayer(40);
		nn->AddInputLayer(2);
		  
		nn->AddNeuralLayer(80, { { 0, 0.5f, {} } , {1, 0.5f, {}} },		{});
		nn->AddNeuralLayer(4, { { 2, 0.0f, {} } },						{});
		nn->AddNeuralLayer(85, {},											{ { 0, 0.5f, {} }, { 1, 0.3f, {}} });
		  
		nn->AddNeuralLayer(3, {},											{ { 2, 0.5f, {} } });
	

		baseGenNN = std::move( nn);

		SwitchToNextGeneration();

		creatureLifeStart = std::chrono::high_resolution_clock::now();

		specimenCount = 0;
		std::unique_ptr<Network> crn = std::make_unique<Network>( allGenNNs [specimenCount].first);
		creature->SetNN( std::move( crn));
	}



	void EvolutionControl::ProcessCreature()
	{

		


		creature->ProcessData();
	}

	void EvolutionControl::SwitchToNextGeneration()
	{

		std::vector<std::pair<Network, std::chrono::steady_clock::duration>> AllGenNNs;
		for( int i = 0; i < GEN_SIZE; i++ )
		{
			AllGenNNs.emplace_back(*baseGenNN, std::chrono::steady_clock::duration::zero());
		}
		
		if( generationCount.has_value() )
		{
			printf( "gen count = %d ", generationCount.value());
			for( auto& nn : AllGenNNs )
			{
				nn.first.MutateWeights(0.15f);
			}
			generationCount.value()++;

		} else {
			for( auto& nn : AllGenNNs )
			{
				nn.first.PopulateWeightMatrices();
			}
			generationCount = 0;
		}


		allGenNNs = std::move( AllGenNNs);

		

	}

	void EvolutionControl::ProcessEvolution()
	{
		// add/remove food/threats if needed
		// add/remove neurons/layers if needed
		//printf(" %d %d ", specimenCount, generationCount.has_value()?generationCount.value():0);

		if( !isCreatureAlive  ) // main logic of applying mutations/ spawning generations begins with death of a creature
		{
			allGenNNs [specimenCount].second = std::chrono::high_resolution_clock::now() - creatureLifeStart;

			if( specimenCount >= allGenNNs.size() )
			{
				assert(false);
				return;
			}

			printf( "%d  ", allGenNNs [specimenCount].second.count() / 1000000000);
			
			specimenCount++;

			if( specimenCount >= allGenNNs.size() ) // spawn generation here
			{

				auto bestResult = std::max_element( allGenNNs.begin(), allGenNNs.end(), [](const std::pair<Network, std::chrono::steady_clock::duration>& o1, 
					const std::pair<Network, std::chrono::steady_clock::duration>& o2)
				{
					return o1.second < o2.second;
				});

				printf( "\n picked time: %I64d ", bestResult->second.count());
				printf( "\ first weights: \n %.2f  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f  %.2f ", 
					bestResult->first.GetNeuralLayer(0)->_FeedingLayers [0].second._weights [0],
					bestResult->first.GetNeuralLayer(0)->_FeedingLayers [0].second._weights [1],
					bestResult->first.GetNeuralLayer(0)->_FeedingLayers [0].second._weights [2],
					bestResult->first.GetNeuralLayer(0)->_FeedingLayers [0].second._weights [3],
					bestResult->first.GetNeuralLayer(0)->_FeedingLayers [0].second._weights [4],
					bestResult->first.GetNeuralLayer(0)->_FeedingLayers [0].second._weights [5],
					bestResult->first.GetNeuralLayer(0)->_FeedingLayers [0].second._weights [6],
					bestResult->first.GetNeuralLayer(0)->_FeedingLayers [0].second._weights [7]
					);
				baseGenNN = std::make_unique<Network>(bestResult->first);//  std::move( bestResult->first);

				SwitchToNextGeneration();

				specimenCount = 0;
				printf( "\n");
			} 

			creatureLifeStart = std::chrono::high_resolution_clock::now();
			isCreatureAlive = true;


			std::unique_ptr<Network> nn = std::make_unique<Network>( allGenNNs [specimenCount].first);
			creature->SetNN( std::move( nn));
			scene->ResetPlayer( 600.0f, 600.0f);
		}

		//ProcessCreature();

	}


	void EvolutionControl::onCreatureDies()
	{
		isCreatureAlive = false;
	}

	void Creature::ProcessData()
	{

		nn->SetInputData(0, player->view.first);
		nn->SetInputData(1, player->view.second);
		nn->SetInputData(2, { player->fed, player->health });

		nn->RunNetwork();
		output = nn->GetOutput();

		/*printf("\n %.2f", player->fed);
		printf("  %.2f", player->health);*/
		//printf(" \n %.1f %.1f", player->worldPos.x, player->worldPos.y);
		
		/*printf("  %.3f", output [0]);
		printf("  %.3f", output [1]);
		printf("  %.3f", output [2]);*/

		player->ChangeSpeed(output [0]);

		player->ChangeAngularSpeed(output [1]);
		player->ChangeAngularSpeed(output [2]);

	}

}
