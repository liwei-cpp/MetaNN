#pragma once

#include <MetaNN/data/facilities/traits.h>
namespace MetaNN
{
// matrices
template<typename TElement, typename TDevice>
class ThreeDArray;

template <typename TElement, typename TDevice>
constexpr bool IsThreeDArray<ThreeDArray<TElement, TDevice>> = true;
}
