#include "environment.h"
#include "../Graphics/graphics.h"


namespace engine
{
	namespace environment
	{


		void Player::Update(std::vector<std::shared_ptr<Object>>& objects, uint16_t timeDiff)
		{
			
			fed -= timeDiff * fedDecreaseRate;

			if( fed <= 0.0f )
			{
				// destroy the player;
				//assert(false);

				cbHolder->onDie(this);
				return;
			}


			auto [newWorldPos, newWorldAngle] = Movable::Move( worldPos, worldAngle, timeDiff);

			std::vector<graphics::Side> newSides = physics::CalcSidesPos( sides, newWorldAngle, newWorldPos);

			for( auto& o : objects )
			{
				if( o.get() == this )
					continue;

				if( physics::DetectCollision(sidesInWorld, newSides, o->sidesInWorld) )
				{
					switch( o->TypeId() )
					{
					
					case ActorTypeId::FoodType:
					{
						//eat this
						fed += 0.2f;
						if( fed > 1.1f )
						{
							fed = 1.1f;
							health -= 0.2f;
							if( health <= 0.0f )
							{
								cbHolder->onDie(this);
								return;
							}
						}

						// and apply new pos/angle/sides;
						Movable::InvertMovement();
						worldAngle = speed.first;
						return;

						break;
					}
					case ActorTypeId::ThreatType:
					{
						health -= 0.25f;
						if( health <= 0.0f )
						{
							cbHolder->onDie(this);
							// destroy the player
							return;
						}
						
						Movable::InvertMovement();
						worldAngle = speed.first;
						return;
						break;
					}

					case ActorTypeId::ObstacleType:
					{

						Movable::InvertMovement();
						worldAngle = speed.first;
						return;
						break;
					}

					case ActorTypeId::BorderType:
					{
						Movable::InvertMovement();
						worldAngle = speed.first;
						return;
						break;
					}

					default:
						assert(false);
						break;
					}


					break;
				}

				
				
			}

			worldAngle = newWorldAngle;
			worldPos = std::move(newWorldPos);

			sidesInWorld = std::move(newSides);


		}

		uint8_t Player::TypeId()
		{
			return ActorTypeId::PlayerType;
		}

		void Threat::Update(std::vector<std::shared_ptr<Object>>& objects, uint16_t timeDiff)
		{
			auto [newWorldPos, newWorldAngle] = Movable::Move(worldPos, worldAngle, timeDiff);
			
			std::vector<graphics::Side> newSides = physics::CalcSidesPos(sides, newWorldAngle, newWorldPos);

			for( auto& o : objects )
			{
				if( o.get() == this )
					continue;

				if( physics::DetectCollision(sidesInWorld, newSides, o->sidesInWorld) )
				{
					switch( o->TypeId() )
					{

					case ActorTypeId::ObstacleType:
						break;

					case ActorTypeId::FoodType:
						break;

					case ActorTypeId::PlayerType:
					{
						if( Player* p = dynamic_cast<Player*>( o.get() ) )
						{
							p->health -= 0.25f;

						} else {

							assert(false);
							return;
						}

						Movable::InvertMovement();
						worldAngle = speed.first;
						return;


						break;
					}

					case ActorTypeId::BorderType:
					{
						Movable::InvertMovement();
						worldAngle = speed.first;
						return;
						break;
					}

					default:
						assert(false);
						break;
					}

				}

				break;


			}

			worldAngle = newWorldAngle;
			worldPos = std::move(newWorldPos);

			sidesInWorld = std::move(newSides);


		}

		uint8_t Threat::TypeId()
		{
			return ActorTypeId::ThreatType;
		}

		void Food::Update(std::vector<std::shared_ptr<Object>>& objects, uint16_t timeDiff)
		{
			std::vector<graphics::Side> newSides = physics::CalcSidesPos(sides, worldAngle, worldPos);
			
			sidesInWorld = std::move(newSides);

			//do nothing, but may be later
		}

		uint8_t Food::TypeId()
		{
			return ActorTypeId::FoodType;
		}

		void Obstacle::Update(std::vector<std::shared_ptr<Object>>& objects, uint16_t timeDiff)
		{
			std::vector<graphics::Side> newSides = physics::CalcSidesPos(sides, worldAngle, worldPos);
			
			sidesInWorld = std::move(newSides);
			// do nothing, but may be later
		}

		uint8_t Obstacle::TypeId()
		{
			return ActorTypeId::ObstacleType;
		}

		void Border::Update(std::vector<std::shared_ptr<Object>>& objects, uint16_t timeDiff)
		{
			std::vector<graphics::Side> newSides = physics::CalcSidesPos(sides, worldAngle, worldPos);
			sidesInWorld = std::move(newSides);
			// do nothing
		}

		uint8_t Border::TypeId()
		{
			return ActorTypeId::BorderType;
		}

		void Scene::CalcAndSetPlayerView(float distToProjPlane, int halfViewField)
		{

			std::vector<graphics::Side> allSidesFromViewerPoV;
			std::vector<int8_t> types;

			for( auto& obj : objects )
			{
				if( obj == player)
					continue;

				for( auto& s : obj->sidesInWorld )
				{
					graphics::Point moved1 = graphics::Move(s.p1, graphics::Point(-player->worldPos.x, -player->worldPos.y));
					graphics::Point moved2 = graphics::Move(s.p2, graphics::Point(-player->worldPos.x, -player->worldPos.y));

					graphics::Point moved_rot1 = graphics::RotVec(moved1, -player->worldAngle);
					graphics::Point moved_rot2 = graphics::RotVec(moved2, -player->worldAngle);

					allSidesFromViewerPoV.emplace_back( moved_rot1, moved_rot2);

					types.emplace_back( obj->TypeId());

				}


			}

			//std::pair<std::vector<float>, std::vector<float>> projPlane = graphics::GetProjectPlane(allSidesFromViewerPoV, types, distToProjPlane, halfViewField);

			player->view = graphics::GetProjectPlane(allSidesFromViewerPoV, types, distToProjPlane, halfViewField);

		}

		void Scene::Update(uint16_t timeDiff)
		{
			for( auto& o : objects )
			{
				o->Update(objects, timeDiff);
			}

			int asdf = 0;
			//player->Update(objects, timeDiff);
		}

		void Scene::MovePlayerTo(float x, float y)
		{
			player->worldPos.x = x;
			player->worldPos.y = y;
		}

		void Scene::addObjectToDeletion(physics::Object* obj)
		{
			deletingObjects.emplace_back( obj);
		}

		void Scene::callOnPlayerDies()
		{
			if( onPlayerDied )
					onPlayerDied();
		}

		void ActorCallbacks::onDie(physics::Object* obj)
		{
			switch( obj->TypeId() )
			{
			case ActorTypeId::PlayerType:
				scene->callOnPlayerDies();
				break;


			default:
				break;
			}


		}

		void ActorCallbacks::onDestroySelf(physics::Object* obj)
		{
			switch( obj->TypeId() )
			{
				scene->callOnPlayerDies();

			default:
				break;
			}


			scene->addObjectToDeletion(obj);
		};


} // namespace environment

} // namespace engine