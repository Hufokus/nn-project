#ifndef _PHYSICS_H
#define _PHYSICS_H

#include <cassert>
#include <vector>
#include "../Graphics/graphics.h"
#include <memory>


namespace engine
{
	namespace physics
	{

		class Object
		{

		public:

			enum class GeomType
			{
				rectangle,
				triangle,
				inverted_rectangle
			};

		public:
			Object(float w, float h, GeomType type, std::pair<float, float> initialPos);
			
		
			std::vector<graphics::Side> sides;
			std::vector<graphics::Side> sidesInWorld;
			graphics::Point worldPos{ 0, 0 };
			float worldAngle{ 0.0f };
			//std::vector<graphics::Side> calculatedSides;
			

		public:
			
			virtual void Update(std::vector<std::shared_ptr<Object>>& objects, uint16_t timeDiff) = 0;
			virtual uint8_t TypeId() = 0;

		};


		class Movable
		{
		public:
			Movable()
			{}

			Movable(float angSpd, std::pair<float, float> spd)
				: angularSpeed(angSpd), speed(std::move(spd))
			{}

			std::pair<graphics::Point, float> Move(const graphics::Point oldWorldPos, const float oldWorldAngle, const float timeDiff);

			void InvertMovement()
			{
				graphics::CompressAngle( speed.first);
				
				/*while( speed.first < 0 )
					speed.first = speed.first + doublePi;


				while( speed.first > doublePi )
					speed.first = speed.first - doublePi;*/

				speed.first -= Pi;
			}

			void ChangeSpeed(float val)
			{
				speed.second += val;
			}

			void ChangeAngularSpeed(float val)
			{
				angularSpeed += val;
			}


		
			std::pair<float /*angle*/, float /*scalar*/> speed{ 0.0f, 0.0f };
			float angularSpeed{ 0.0f };
			float envCharge{ 0.0f };
		};

		std::vector<graphics::Side> CalcSidesPos(std::vector<graphics::Side>& sides, float worldAngle, graphics::Point worldPos);

		bool DetectCollision( const std::vector<graphics::Side>& oldSides1, const std::vector<graphics::Side>& newSides1,
			const std::vector<graphics::Side>& sides2 );



	} // namespace physics

} // namespace engine


#endif // _PHYSICS_H
