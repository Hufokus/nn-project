#include "matrix_supp.h"


namespace mx
{

	void InsertRow(std::vector<float>& vec, const uint16_t row, const float value, const uint16_t count)
	{
		auto it = vec.begin() + row * count;
		vec.insert(it, count, value);
	}


	void InsertColumn(std::vector<float>& vec, const uint16_t pos, const float value, const uint16_t columnsCount)
	{

		for( uint16_t c = pos; c < vec.size(); c += columnsCount )
		{
			auto it = vec.begin() + c;
			vec.insert(it, value);
		}

	}
}