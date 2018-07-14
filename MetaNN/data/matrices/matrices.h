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

template <typename TElem, typename TDevice>
auto SubMatrix(const Matrix<TElem, TDevice>& input,
               size_t p_rowB, size_t p_rowE, size_t p_colB, size_t p_colE)
{
    auto res = input.SubMatrix2(p_rowB, p_rowE, p_colB, p_colE);
    return Matrix<TElem, TDevice>(std::move(res));
}

}
