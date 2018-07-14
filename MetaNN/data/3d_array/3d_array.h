#pragma once

#include <MetaNN/data/facilities/traits.h>
namespace MetaNN
{
// matrices
template<typename TElement, typename TDevice>
class ThreeDArray;

template <typename TElem, typename TDevice>
struct DataCategory_<ThreeDArray<TElem, TDevice>>
{
    using type = CategoryTags::ThreeDArray;
};
}
