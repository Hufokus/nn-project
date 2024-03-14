

#include "graphics.h"
#include <cmath>


#include <utility>
#include <math.h>
#include <vector>
#include <utility>
#include <set>
#include <optional>


namespace engine
{
	namespace graphics
	{

		void CompressAngle(float& angle)
		{

			while( angle < 0 )
				angle = angle + doublePi;


			while( angle > doublePi )
				angle = angle - doublePi;
		}

		float AddAngle(const float a1, const float a2)
		{
			float res{ a1 + a2 };
			CompressAngle( res);
			return res;
		}

		std::pair<float, float> RotVec(const std::pair<float, float> pos, const float angle)
		{
			float a{ angle };

			CompressAngle( a);
			

			float s = sin(a);
			float c = cos(a);

			float x = c * pos.first - s * pos.second;
			float y = s * pos.first + c * pos.second;

			return { x, y };
		}


		Point RotVec(const Point pos, const float angle)
		{
			float a{ angle };
			
			CompressAngle( a);
			

			float s = sin(a);
			float c = cos(a);

			float x = c * pos.x - s * pos.y;
			float y = s * pos.x + c * pos.y;

			return { x, y };
		}

		Point Move(const Point pos, const Point vec)
		{
			return { pos.x + vec.x, pos.y + vec.y };
		}

		Side CalcWorld(const Side side, float angle, Point moveVec)
		{
			Side s = { RotVec(side.p1, angle), RotVec(side.p2, angle) };

			s.p1 = Move(s.p1, moveVec);
			s.p2 = Move(s.p2, moveVec);

			return s;

			//return side;
		}

		int proj(std::pair<float, float> coords, float dist)
		{
			float res = coords.first * dist / coords.second;
			//float res = coords.first / coords.second;
			return std::round(res);
		}

		std::pair<std::vector<float>, std::vector<float>> GetProjectPlane(std::vector<Side> sides, std::vector<int8_t> types, float dist, int halfWidth)
		{
			std::pair<std::vector<float>, std::vector<float>> result;
			result.first.assign(halfWidth * 2, 0.0f);
			result.second.assign(halfWidth * 2, 9999.0f);

			int k = -1;

			for( auto& s : sides )
			{
				k++;

				if( s.p1.y < dist && s.p2.y < dist)
					continue;

				float Ny = s.p2.x - s.p1.x;
				float Nx = s.p1.y - s.p2.y;


				float dot = s.p1.x * Nx + s.p1.y * Ny;

				if( dot >= 0 )
					continue;

				int x1{ 0 };

				std::optional<Point> under{};
				std::optional<Point> above{};

				if( s.p1.y < dist )
				{
					under = s.p1;
					above = s.p2;
				}

				if( s.p2.y < dist )
				{
					under = s.p2;
					above = s.p1;
				}

				if( under )
				{
					float a = above.value().y - dist;
					float b = std::abs( above.value().y - under.value().y);
					float c = std::abs( under.value().x - above.value().x);
					/*float a = s.p2.y - dist;
					float b = s.p2.y + dist - s.p1.y;
					float c = s.p1.x - s.p2.x;

					float dx = c * a / b;
					float x1 = dx + s.p2.x;*/

					float dx = c * a / b;

					above.value().x > under.value().x ? x1 = above.value().x - dx : x1 = above.value().x + dx;
					s.p2 = above.value();
					//x1 = dx + above.value().x;

				} else
					x1 = proj({ (float)s.p1.x, (float)s.p1.y }, dist);

				int y = std::round(dist);
				float a = std::pow(x1 - s.p1.x, 2);
				float b = std::pow(y - s.p1.y, 2);
				float depth1 = sqrt(a + b /*(x1 - s.p1.x)^2 + (y - s.p1.y)^2*/);
				if( depth1 < 1.0f )
						depth1 = 1.0f;

				//x1 += halfWidth;

				x1 = halfWidth - x1;
				/*if( x1 >= 0 && x1 < halfWidth * 2 )
				{
					if( depth1 < result [x1].second )
							result [x1] = std::make_pair(255, depth1);

				}*/

				int x2 = proj({ s.p2.x, s.p2.y }, dist);
				y = std::round(dist);

				float depth2 = sqrt(std::pow(x2 - s.p2.x, 2) + std::pow(y - s.p2.y, 2));
				if( depth2 < 1.0f )
						depth2 = 1.0f;
				//x2 += halfWidth;
				x2 = halfWidth - x2;
				/*if( x2 >= 0 && x2 < halfWidth * 2 )
				{

					if( depth2 < result [x2].second )
							result [x2] = std::make_pair(255, depth2);

				}*/

				std::pair<int, float> p1, p2;
				p1.first = x1;
				p1.second = depth1;

				p2.first = x2;
				p2.second = depth2;


				if( p1.first > p2.first )
					std::swap(p1, p2);

				int fullWidth = halfWidth * 2;
				if( ( p1.first < 0 && p2.first < 0 ) || ( p1.first > fullWidth && p2.first > fullWidth ) )
					continue;

				/*if( x1 > x2)
					std::swap( x1, x2);*/

					/*if( depth1 > depth2)
						std::swap( depth1, depth2);*/

						/*int diff = x2 - x1;
						float step = (depth2 - depth1) / diff;*/

				int diff = p2.first - p1.first;
				float step = std::abs(( p2.second - p1.second ) / diff);

				if( p1.second > p2.second )
					step = -step;

				/*if( result [x1].second > result [x2].second)
					step = -step;*/
				float curDepths = p1.second;

				for( int i = p1.first; i < p2.first; i++ )
				{
					if( i >= fullWidth )
						break;

					if( i < 0 )
					{

						curDepths = curDepths + step;
						continue;
					}

					//float curDepth = result [i - 1].second + step;
					//float curDepth = prev.second + step;

					if( curDepths <  result.second [i] )
					{
						result.first [i] = types [k]; /*255;*/
						result.second [i] = curDepths;
					}

					curDepths = curDepths + step;

				}

				//if( p2.first < fullWidth )
				//{
				//	result.first [p2.first] = types [k]; /*255;*/
				//	result.second [p2.first] = p2.second;
				//}



				int asdfffs = 0;

			}


			for( int i = 0; i < result.first.size(); i++)
			{
				if( result.first [i] > 0.01f )
						result.first [i] = result.first [i]/* / 10.0f*/;

				result.second [i] = 1 / result.second [i] * 100.0f;
				if( result.second[i] > 1.0f)
						result.second [i] = 1.0f;
			}

			return result;
		}


	} // namespace graphics

} // namespace engine