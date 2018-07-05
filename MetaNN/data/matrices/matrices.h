#pragma once

#include <MetaNN/data/facilities/traits.h>
namespace MetaNN
{
// matrices
template<typename TElement, typename TDevice>
class Matrix;

template <typename TElement, typename TDevice>
constexpr bool IsMatrix<Matrix<TElement, TDevice>> = true;
}
