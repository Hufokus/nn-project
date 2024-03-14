
#include "physics.h"

#include <cmath>

namespace engine
{
	namespace physics
	{


		std::vector<graphics::Side> CalcSidesPos(std::vector<graphics::Side>& sides, float worldAngle, graphics::Point worldPos)
		{
			std::vector<graphics::Side> newSides(sides.size());

			for( int i = 0; i < sides.size(); i++ )
				newSides [i] = graphics::CalcWorld( sides [i], worldAngle, worldPos);

			return newSides;

		}

		bool DetectCollision(const std::vector<graphics::Side>& oldSides1, const std::vector<graphics::Side>& newSides1, 
			const std::vector<graphics::Side>& sides2)
		{

			// put optimization here


			for( int i = 0; i < oldSides1.size(); i++ )
			{
				for( const auto& s2 : sides2 )
				{

					float x1 = newSides1 [i].p1.x;
					float y1 = newSides1 [i].p1.y;

					float x2 = newSides1 [i].p2.x;
					float y2 = newSides1 [i].p2.y;

					float x3 = s2.p1.x;
					float y3 = s2.p1.y;

					float x4 = s2.p2.x;
					float y4 = s2.p2.y;


					float u1 = ( x1 - x3 ) * ( y3 - y4 ) - ( y1 - y3 ) * ( x3 - x4 );
					float d1 = ( x1 - x2 ) * ( y3 - y4 ) - ( y1 - y2 ) * ( x3 - x4 );

					bool itrc1 = abs(u1) < abs(d1) && ( ( u1 > 0 && d1 > 0 ) || ( u1 < 0 && d1 < 0 ) );


					u1 = ( x1 - x3 ) * ( y1 - y2 ) - ( y1 - y3 ) * ( x1 - x2 );
					d1 = ( x1 - x2 ) * ( y3 - y4 ) - ( y1 - y2 ) * ( x3- x4 );

					bool itrc2 = abs(u1) < abs(d1) && ( ( u1 > 0 && d1 > 0 ) || ( u1 < 0 && d1 < 0 ) );

					if( itrc1 && itrc2 )
						return true;







					 x1 = oldSides1 [i].p1.x;
					 y1 = oldSides1 [i].p1.y;

					 x2 = newSides1 [i].p1.x;
					 y2 = newSides1 [i].p1.y;

					 x3 = s2.p1.x;
					 y3 = s2.p1.y;

					 x4 = s2.p2.x;
					 y4 = s2.p2.y;

					
					 u1 = ( x1 - x3) * ( y3 - y4) - ( y1 - y3) * (x3 - x4);
					 d1 = (x1 - x2) * (y3 - y4) - (y1 - y2) * ( x3 - x4);

					 itrc1 = abs( u1) < abs( d1) && (( u1 > 0 && d1 > 0) || ( u1 < 0 && d1 < 0));
				

					u1 = (x1 - x3) * (y1 - y2) - (y1 - y3) * ( x1 - x2);
					d1 = ( x1 - x2) * (y3 - y4) - ( y1 - y2) * ( x3- x4);

					 itrc2 = abs(u1) < abs(d1) && ( ( u1 > 0 && d1 > 0 ) || ( u1 < 0 && d1 < 0 ) );
						
					if( itrc1 && itrc2)
						return true;


				}
			}

			return false;
		}

		std::pair<graphics::Point, float> Movable::Move(const graphics::Point oldWorldPos, const float oldWorldAngle, const float timeDiff)
		{
			std::pair<float, float> speedVector{ 0.0f, 1.0f };

			angularSpeed -= envCharge * timeDiff * 0.01f/*0.00001f*/; // affect angularV with environment
			if( angularSpeed < 0.0f )
					angularSpeed = 0.0f;
			if( angularSpeed > 100.0f )
					angularSpeed = 100.0f;

			float newWorldAngle = oldWorldAngle;

			newWorldAngle/*speed.first*/ += angularSpeed * timeDiff * 0.01f; // calc linearV vector rotation accordingly to angularV
				
			graphics::CompressAngle(newWorldAngle);

			/*float pi2 = 3.14159f * 2.0f;*/
			//if( abs(oldWorldAngle/*speed.first*/) > doublePi )
			//	oldWorldAngle/*speed.first*/ = 0.0f;

			speedVector = graphics::RotVec(speedVector, newWorldAngle/*speed.first*/);

			speed.second -= envCharge *  timeDiff * 0.01f/*0.0001f*/; // affect linearV with environment



			if( speed.second < 0.0f)
					speed.second = 0.0f;
			if( speed.second > 100.0f)
					speed.second = 1000.0f;

			speedVector.first *= speed.second * timeDiff * 0.00001f; // calc new world pos accordingly to linearV
			speedVector.second *= speed.second * timeDiff * 0.00001f;



			graphics::Point newWorld{ oldWorldPos.x + speedVector.first,  oldWorldPos.y + speedVector.second };

			

			return { newWorld, newWorldAngle };
		}

		Object::Object(float w, float h, GeomType type, std::pair<float, float> initialPos) :
			worldPos(initialPos.first, initialPos.second)
		{


			switch( type )
			{
			case GeomType::rectangle:
			{

				sides.emplace_back(graphics::Point{ -( w/2 ), -( h/2 ) }, graphics::Point{ -( w/2 ), h/2 });
				sides.emplace_back(graphics::Point{ -( w/2 ), h/2 }, graphics::Point{ w/2, h/2 });
				sides.emplace_back(graphics::Point{ w/2, h/2 }, graphics::Point{ w/2, -( h/2 ) });
				sides.emplace_back(graphics::Point{ w/2, -( h/2 ) }, graphics::Point{ -( w/2 ), -( h/2 ) });
				sidesInWorld = CalcSidesPos(sides, worldAngle, worldPos);
				break;
			}

			case GeomType::triangle:
			{
				sides.emplace_back(graphics::Point{ -( w/2 ), -( h/2 ) }, graphics::Point{ 0.0f, h/2 });
				sides.emplace_back(graphics::Point{ 0.0f, h/2 }, graphics::Point{ w/2, -h/2 });
				sides.emplace_back(graphics::Point{ w/2, -h/2 }, graphics::Point{ -( w/2 ), -( h/2 ) });
				sidesInWorld = CalcSidesPos(sides, worldAngle, worldPos);
				break;
			}

			case GeomType::inverted_rectangle:
			{
				sides.emplace_back(graphics::Point{ w/2, -( h/2 ) }, graphics::Point{ -( w/2 ), -( h/2 ) });
				sides.emplace_back(graphics::Point{ w/2, h/2 }, graphics::Point{ w/2, -( h/2 ) });
				sides.emplace_back(graphics::Point{ -( w/2 ), h/2 }, graphics::Point{ w/2, h/2 });
				sides.emplace_back(graphics::Point{ -( w/2 ), -( h/2 ) }, graphics::Point{ -( w/2 ), h/2 });
				sidesInWorld = CalcSidesPos(sides, worldAngle, worldPos);
				break;
			}

			default:
				assert(false);
				break;


			}

		}

} // namespace graphics

} // namespace engine