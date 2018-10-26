#pragma once

#include <MetaNN/data/facilities/traits.h>
namespace MetaNN
{
// matrices
template<typename TElement, typename TDevice>
class Matrix;

template <typename TElem, typename TDevice>
struct DataCategory_<Matrix<TElem, TDevice>>
{
    using type = CategoryTags::Matrix;
};
}
