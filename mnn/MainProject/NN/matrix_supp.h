#ifndef  _MATRIX_SUPP_H
#define _MATRIX_SUPP_H

#include <vector>

namespace mx
{

	void InsertRow(std::vector<float>& vec, const uint16_t row, const float value, const uint16_t count);
	void InsertColumn(std::vector<float>& vec, const uint16_t pos, const float value, const uint16_t columnsCount);

}


#endif // ! _MATRIX_SUPP_H

