#ifndef _ENVIRONMENT_H
#define _ENVIRONMENT_H

#include "../Physics/physics.h"

#include <chrono>
#include <functional>



namespace engine
{
	namespace environment
	{


		enum ActorTypeId
		{
			PlayerType = 0,
			ObstacleType = 1,
			ThreatType = 2, 
			FoodType = 3,
			BorderType = 4
		};

		class Scene;
		class ActorCallbacks;

		class ActorCallbacks
		{
		public:
			ActorCallbacks(Scene* scene)
				: scene(scene)
			{}

			void onDie(physics::Object* obj);
			void onDestroySelf(physics::Object* obj);
			
		private:
			Scene* scene;
		};
	


		class Player : public physics::Object, public physics::Movable
		{
		public: 
			Player(float w, float h, GeomType type, std::pair<float, float> initialPos, std::shared_ptr<ActorCallbacks> cbHolder) :
				cbHolder( std::move( cbHolder)),
				Object(w, h, type, std::move(initialPos))
			{
			}


			float fed{ 1.0f };
			float health{ 1.0f };
			
			float healthRestoreRate{ 0.0000001f };
			float fedDecreaseRate{ 0.00000015f };

			friend class Threat;
			friend class Food;

		public:
			std::pair<std::vector<float>, std::vector<float>> view;
			
			std::shared_ptr<ActorCallbacks> cbHolder{};
			virtual void Update( std::vector<std::shared_ptr<Object>>& objects, uint16_t timeDiff) override;
			virtual uint8_t TypeId() override;

			
		};


		class Threat : public physics::Object, physics::Movable
		{
		public:
			Threat(float w, float h, GeomType type, std::pair<float, float> initialPos, std::shared_ptr<ActorCallbacks> cbHolder) :
				cbHolder(std::move(cbHolder)), 
				Object(w, h, type, std::move( initialPos))
			{}

		public:

			virtual void Update(std::vector<std::shared_ptr<Object>>& objects, uint16_t timeDiff) override;
			virtual uint8_t TypeId() override;

		private:
			std::shared_ptr<ActorCallbacks> cbHolder{};
		};



		class Food : public physics::Object/*, physics::Movable*/
		{
		public:

			Food(float w, float h, GeomType type, std::pair<float, float> initialPos, std::shared_ptr<ActorCallbacks> cbHolder) :
				cbHolder(std::move(cbHolder)),
				Object( w, h, type, std::move( initialPos))
			{}


		public:
			virtual void Update(std::vector<std::shared_ptr<Object>>& objects, uint16_t timeDiff) override;
			virtual uint8_t TypeId() override;

		private:
			std::shared_ptr<ActorCallbacks> cbHolder{};
		};



		class Obstacle : public physics::Object
		{
		public:
			Obstacle(float w, float h, GeomType type, std::pair<float, float> initialPos, std::shared_ptr<ActorCallbacks> cbHolder) :
				cbHolder(std::move(cbHolder)), 
				Object(w, h, type, std::move(initialPos))
			{}

		public:
			virtual void Update(std::vector<std::shared_ptr<Object>>& objects, uint16_t timeDiff) override;
			virtual uint8_t TypeId() override;

		private:
			std::shared_ptr<ActorCallbacks> cbHolder{};
		};




		class Border : public physics::Object
		{
		public:
			Border(float w, float h, GeomType type, std::pair<float, float> initialPos, std::shared_ptr<ActorCallbacks> cbHolder) :
				cbHolder(std::move(cbHolder)),
				Object(w, h, type, std::move(initialPos))
			{}
			virtual void Update(std::vector<std::shared_ptr<Object>>& objects, uint16_t timeDiff) override;
			virtual uint8_t TypeId() override;

		private:
			std::shared_ptr<ActorCallbacks> cbHolder{};
		};


		class Scene
		{
		public:
			Scene( std::function<void()> onPlayerDied)
				: onPlayerDied(std::move(onPlayerDied))
			{
				actorCallbacks = std::make_shared<ActorCallbacks>( this);
			}

			std::shared_ptr<Player> AddPlayer(float w, float h, std::pair<float, float> initialPos)
			{
				player = std::make_shared<Player>(w, h, physics::Object::GeomType::triangle, std::move(initialPos), actorCallbacks);
				objects.emplace_back(player);
				return player;
			}

			void AddFood(float w, float h, std::pair<float, float> initialPos)
			{
				objects.emplace_back(std::make_shared<Food>(w, h, physics::Object::GeomType::rectangle, std::move(initialPos), actorCallbacks));
			}

			void AddThreat(float w, float h, std::pair<float, float> initialPos)
			{
				objects.emplace_back(std::make_shared<Threat>(w, h, physics::Object::GeomType::rectangle, std::move(initialPos), actorCallbacks));
			}

			void AddObstacle(float w, float h, std::pair<float, float> initialPos)
			{
				objects.emplace_back(std::make_shared<Obstacle>(w, h, physics::Object::GeomType::rectangle, std::move(initialPos), actorCallbacks));
			}

			void AddBorders(float w, float h, std::pair<float, float> initialPos)
			{
				std::pair<float, float> pos;
				pos.first = initialPos.first;
				pos.second = initialPos.second - h / 2 - 10;
				objects.emplace_back(std::make_shared<Border>(w, 20, physics::Object::GeomType::rectangle, std::move(pos), actorCallbacks));

				pos.first = initialPos.first + w / 2 + 10;
				pos.second = initialPos.second;
				objects.emplace_back(std::make_shared<Border>(20, h, physics::Object::GeomType::rectangle, std::move(pos), actorCallbacks));

				pos.first = initialPos.first;
				pos.second = initialPos.second + h / 2 + 10;
				objects.emplace_back(std::make_shared<Border>(w, 20, physics::Object::GeomType::rectangle, std::move(pos), actorCallbacks));

				pos.first = initialPos.first - w / 2 - 10;
				pos.second = initialPos.second;
				objects.emplace_back(std::make_shared<Border>(20, h, physics::Object::GeomType::rectangle, std::move(pos), actorCallbacks));

			}

			/*void AddObject(ActorTypeId type, float w, float h, std::pair<float, float> initialPos)
			{
				switch( type )
				{
				case ActorTypeId::Player:
					objects.emplace_back(std::make_shared<physics::Object>(w, h, physics::Object::GeomType::triangle, std::move(initialPos)));
					break;
				case ActorTypeId::Food:
				case ActorTypeId::Obstacle:
				case ActorTypeId::Threat:
					objects.emplace_back(std::make_shared<physics::Object>(w, h, physics::Object::GeomType::rectangle, std::move(initialPos)));
					break;

				default:
					break;
				}


			}*/

			void ResetPlayer(float x, float y)
			{
				MovePlayerTo( x, y);
				player->fed = 1.0f;
				player->health = 1.0f;
				player->angularSpeed = 0.0f;
				player->speed.first = 0.0f;
				player->speed.second = 0.0f;
				player->worldAngle = 0.0f;
				player->sidesInWorld = physics::CalcSidesPos(player->sides, player->worldAngle, player->worldPos);
			}
			void MovePlayerTo( float x, float y);
			void CalcAndSetPlayerView(float distToProjPlane, int halfViewField);
			void Update(uint16_t timeDiff);

			void addObjectToDeletion( physics::Object* obj);
			void callOnPlayerDies();

			std::function<void()> onPlayerDied{};
			
			std::shared_ptr<ActorCallbacks> actorCallbacks;
			std::vector<physics::Object*> deletingObjects;
			std::vector<std::shared_ptr<physics::Object>> objects;
			std::shared_ptr<Player> player;

			friend class ActorCallbacks;
		};


	} // namespace environment

} // namespace engine



#endif /*_ENVIRONMENT_H*/
