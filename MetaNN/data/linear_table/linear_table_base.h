#pragma once

#include <MetaNN/data/facilities/shape.h>

namespace MetaNN
{
namespace NSLinearTable
{
template <typename TCardinal>
size_t WrapperDim(const Shape_<CategoryTags::Batch<TCardinal>>& shape)
{
    return shape.BatchNum();
}

template <typename TCardinal>
size_t WrapperDim(const Shape_<CategoryTags::Sequence<TCardinal>>& shape)
{
    return shape.Length();
}
}

template<typename TElement, typename TDevice, typename TCategory>
class LinearTable;
}