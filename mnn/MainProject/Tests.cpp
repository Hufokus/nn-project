
#include <chrono>

//#include "Network.h"
//#include "cuda_calc.h"

#include "../ExternalLibs/sdl/include/SDL.h"
#include "Evolution/Evolution.h"

//#include "Engine/Physics/physics.h"

//#include " "sdl/include/SDL.h"


//
//#include "VirtualEnvironment/Creature.h"
//#include "VirtualEnvironment/graphic_calcs.h"

int main(int argc, char** args)
{


	//graphics::Side l1{ {-10, -30}, {-10, 30 } };
	//graphics::Side l2{ {-10, 30}, {10, 30 } };
	//graphics::Side l3{ {10, 30}, {10, -30 } };
	//graphics::Side l4{ {10, -30 }, {-10, -30} };

	//std::shared_ptr<venv::Object> obj{ std::make_shared<venv::Object>( std::vector<graphics::Side>{ l1, l2, l3, l4 } )};

	//obj->worldPos.x = 300;
	//obj->worldPos.y = 120;
	//obj->worldAngle = 1.0f;
	//obj->speed.first = 1.52f;
	//obj->speed.second = 0.0009f;
	//std::vector<std::shared_ptr<venv::Object>> objs{ std::move(obj)};


	//graphics::Side l11{ {-30, -30}, {-30, 30 } };
	//graphics::Side l12{ {-30, 30}, {30, 30 } };
	//graphics::Side l13{ {30, 30}, {30, -30 } };
	//graphics::Side l14{ {30, -30 }, {-30, -30} };


	//std::shared_ptr<venv::Object> obj1{ std::make_shared<venv::Object>(std::vector<graphics::Side>{ l11, l12, l13, l14 }) };
	//obj1->worldPos.x = 180;
	//obj1->worldPos.y = 150;
	//obj1->worldAngle = 0.8f;
	//	
	//objs.emplace_back( std::move( obj1));
	//int asdf = 0;


	//graphics::Side c1{ {0, 3}, {2, -3 } };
	//graphics::Side c2{ {2, -3}, {-2, -3 } };
	//graphics::Side c3{ {-2, -3}, {0, 3 } };


	////venv::Creature Creature{ {c1, c2, c3} };

	//int halfViewField = 20;
	//float distToProjPlane = 10.0f;

	//std::shared_ptr<venv::Object> Creature = std::make_shared<venv::Creature>(std::vector<graphics::Side>{c1, c2, c3});

	//Creature->worldPos.x = 180;
	//Creature->worldPos.y = 60;

	//Creature->worldAngle = 0.0f;

	
	evolution::EvolutionControl ec;
	engine::environment::Scene sc{ [&]() {
		ec.onCreatureDies();
	} };
	ec.SetScene(&sc);
	ec.Init();

	if( SDL_Init(SDL_INIT_VIDEO) == 0 ) {
		SDL_Window* window = NULL;
		SDL_Renderer* renderer = NULL;

		if( SDL_CreateWindowAndRenderer(1250, 1000, 0, &window, &renderer) == 0 ) {
			SDL_bool done = SDL_FALSE;

			auto prevTime = std::chrono::high_resolution_clock::now();
			while( !done ) {
				SDL_Event event;

				SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
				SDL_RenderClear(renderer);

				SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);


				//objs.push_back(Creature);



				auto now = std::chrono::high_resolution_clock::now();
				auto timeDiff = now - prevTime;
				prevTime = now;
				sc.Update(std::chrono::duration_cast<std::chrono::microseconds>( timeDiff ).count());


				for( const auto& o : sc.objects )
				{
					switch( o->TypeId() )
					{
					case engine::environment::ActorTypeId::FoodType:
						SDL_SetRenderDrawColor(renderer, 0, 255, 0, SDL_ALPHA_OPAQUE);
						break;

					case engine::environment::ActorTypeId::ObstacleType:
						SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
						break;
					case engine::environment::ActorTypeId::ThreatType :
						SDL_SetRenderDrawColor(renderer, 255, 0, 0, SDL_ALPHA_OPAQUE);
						break;
					case engine::environment::ActorTypeId::PlayerType:
						SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
						break;

					case engine::environment::ActorTypeId::BorderType:
						SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
						break;

					default:
						assert(false);
						break;

					}
					for( const auto& s : o->sidesInWorld )
					{
						/*if( o->TypeId() ==  engine::environment::ActorTypeId::PlayerType)
							printf( "\ncoordY = %.3f", s.p1.y);*/
						SDL_RenderDrawLine(renderer, (int)s.p1.x, (int)s.p1.y, (int)s.p2.x, (int)s.p2.y);
					}

				}
				int halfViewField = 20;
				float distToProjPlane = 10.0f;
				sc.CalcAndSetPlayerView(distToProjPlane, halfViewField);
				ec.ProcessCreature();
				ec.ProcessEvolution();
				

				//venv::CalcObjectsWorldPos( objs, std::chrono::duration_cast<std::chrono::microseconds>( timeDiff).count());
				//
				//uint16_t sideCount{ 0 };
				//for( const auto obj : objs )
				//{
				//	for( const auto& s : obj->calculatedSides)
				//	{
				//		
				//		SDL_RenderDrawLine(renderer,  (int)std::round( s.p1.x), (int)std::round( s.p1.y), (int)std::round( s.p2.x), (int)std::round( s.p2.y));

				//		sideCount++;
				//	}

				//	
				//}

				//objs.pop_back();
				//////////////////////////////

				//std::vector<graphics::Side> allSidesFromViewerPoV( sideCount);

				//for( const auto obj : objs )
				//{
				//	for( const auto& s : obj->calculatedSides )
				//	{
				//		allSidesFromViewerPoV.emplace_back(
				//			graphics::Rot( graphics::Move( s.p1, graphics::Point( -Creature->worldPos.x, -Creature->worldPos.y)), -Creature->worldAngle),
				//			graphics::Rot( graphics::Move( s.p2, graphics::Point( -Creature->worldPos.x, -Creature->worldPos.y)), -Creature->worldAngle)
				//		);
				//		
				//		
				//	}


				//}

				std::pair<std::vector<float>, std::vector<float>> projPlane = sc.player->view;

				//std::vector<int> pxls( projPlane.first.size());
				//std::vector<float> dists( projPlane.second.size());
				//for( int i = 0; i < projPlane.first.size(); i++)
				//{
				//	pxls [i] = projPlane.first [i];
				//	dists [i] = projPlane.second [i];
				//}

				int x = 50;
				SDL_RenderDrawLine( renderer, x, 380, x, 420);
				SDL_RenderDrawLine(renderer, x + halfViewField * 2 * 4, 380, x + halfViewField * 2 * 4, 420);
				x++;
				for( int i = 0; i < projPlane.first.size(); i++ )
				{

				/*	Uint8 c1 = 0;
					Uint8 c2 = 0;
					Uint8 c3 = 0;*/

					/*if( projPlane.second [i] < 125 )
					{
						c1 = 255 - ( projPlane.second [i]) * 2;
						c2 = 255 - ( projPlane.second [i] ) * 2;
						c3 = 255 - ( projPlane.second [i] ) * 2;
					}*/
					//if( projPlane.first [i] > 0.01f )
					//{
					//	projPlane.first [i] *= 10.0f;
					//	//projPlane.first [i];
					//}


					float c1 = std::abs( projPlane.second [i] * 80.0f / 100.0f); /*1 - 1 / projPlane.second [i];*//* * 0.0283f;*/

					//float 
					//float fade = projPlane.second [i] * 0.0283f;
					//float d = 1 - 1/ projPlane.second [i];


					/*if( projPlane.second [i]  >= 1.0f )
						SDL_SetRenderDrawColor(renderer, , 0, 0, SDL_ALPHA_OPAQUE);*/
					if( projPlane.first [i] < 1.0f)
						SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
					else if( projPlane.first [i] >= 1.0f && projPlane.first [i] < 2.0f )
						SDL_SetRenderDrawColor(renderer, (int)( 255 * c1), (int)( 255 * c1 ), (int)( 255 * c1 ), SDL_ALPHA_OPAQUE);
					else if( projPlane.first [i] >= 2.0f && projPlane.first [i] < 3.0f )
						SDL_SetRenderDrawColor(renderer, (int)( 255 * c1 ), 0, 0, SDL_ALPHA_OPAQUE);
					else if( projPlane.first [i] >= 3.0f && projPlane.first [i] < 4.0f )
						SDL_SetRenderDrawColor(renderer, 0, (int)( 255 * c1 ), 0, SDL_ALPHA_OPAQUE);
					else if( projPlane.first [i] >= 4.0f && projPlane.first [i] < 5.0f )
						SDL_SetRenderDrawColor(renderer, (int)( 255 * c1 ), (int)( 255 * c1 ), (int)( 255 * c1 ), SDL_ALPHA_OPAQUE);
								


					int rectWidth = 4;
					
					int y = 400;
					SDL_Rect r;
					r.h = rectWidth;
					r.w = rectWidth;
					r.x = x;
					r.y = y;
					SDL_RenderFillRect( renderer, &r);

					x += rectWidth;
				}

				/////////////////////


				SDL_RenderPresent(renderer);

				

				while( SDL_PollEvent(&event) ) {
					if( event.type == SDL_QUIT ) {
						done = SDL_TRUE;
					}
				}
			}
		}

		if( renderer ) {
			SDL_DestroyRenderer(renderer);
		}
		if( window ) {
			SDL_DestroyWindow(window);
		}
	}
	SDL_Quit();
	return 0;


	//MatrixMul();
	Network nn;
	nn.AddInputLayer(10);
	nn.AddInputLayer(15);

	std::vector<float> tstData0 { 
		0.1f, 
		0.2f, 
		0.3f, 
		0.4f, 
		0.2f, 
		0.6f, 
		0.7f, 
		0.2f, 
		0.25f, 
		0.4f
	};

	std::vector<float> tstData1{
		0.9f,
		0.3f,
		0.45f,
		0.6f,
		0.1f,
		0.3f,
		0.4f,
		0.3f,
		0.5f,
		0.5f,
		0.6f,
		0.23f,
		0.31f,
		0.51f,
		0.25f
	};

	nn.SetInputData( 0, std::move( tstData0));
	nn.SetInputData( 1, std::move( tstData1));
	//======
	nn.AddNeuralLayer(3, { {0, 0.5f, {} } }, {});
	nn.AddNeuralLayer(7, {}, { { 0, 0.3f, {} } });

	nn.AddNeuralLayer(9, { { 1, 0.4f, {} } }, {});
	nn.AddNeuralLayer(11, {}, { {2, 0.3f, {}} });

	nn.AddNeuralLayer(9, {}, { { 3, 0.7f, 0.9f} , {1, 0.6f, {} } });
	//=======
	nn.AddNeuron( 0, 2);

	int asd3f = 0;

	nn.RunNetwork();

	return 0;
}