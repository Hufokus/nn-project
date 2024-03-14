#ifndef _ENGINE_BASE_H
#define _ENGINE_BASE_H


#include <utility>
#include <math.h>
#include <vector>
#include <utility>
#include <set>

constexpr float Pi = 3.1415926535;
constexpr float doublePi = 3.1415926535 * 2;

namespace engine
{

	namespace graphics
	{

		class Point
		{
		public:
			Point()
			{}

			Point(float x, float y)
				: x(x), y(y)
			{}

			float x{ 0 };
			float y{ 0 };

			Point& operator+=(const Point& rhs)
			{
				this->x += rhs.x;
				this->y += rhs.y;
				return *this;
			}
		};

		class Side
		{
		public:
			Side()
			{}

			Side(Point p1, Point p2)
				: p1(p1), p2(p2)
			{}

			Point p1;
			Point p2;
		};

		void CompressAngle(float& angle);
		Point RotVec(const Point, const float angle);
		float AddAngle(const float a1, const float a2);
		std::pair<float, float> RotVec(const std::pair<float, float> pos, const float angle);
		Point Move(const Point pos, const Point vec);
		Side CalcWorld(const Side side, float angle, Point moveVec);
		std::pair<std::vector<float>, std::vector<float>> GetProjectPlane(std::vector<Side> sides, std::vector<int8_t> types, float dist, int halfWidth);



	} // namespace graphics

} // namespace engine


#endif // !_ENDING_BASE_H

