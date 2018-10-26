#pragma once

#include <MetaNN/data/facilities/traits.h>
namespace MetaNN
{
// matrices
template<typename TElement, typename TDevice>
class Vector;

template <typename TElem, typename TDevice>
struct DataCategory_<Vector<TElem, TDevice>>
{
    using type = CategoryTags::Matrix;
};
}
