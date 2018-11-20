#pragma once

#include <MetaNN/data/linear_table/linear_table_pack.h>

namespace MetaNN
{
template<typename TElem, typename TDevice>
using BatchScalar = StaticArray<TElem, TDevice, CategoryTags::Batch, CategoryTags::Scalar>;

template<typename TElem, typename TDevice>
using BatchMatrix = StaticArray<TElem, TDevice, CategoryTags::Batch, CategoryTags::Matrix>;

template<typename TElem, typename TDevice>
using BatchThreeDArray = StaticArray<TElem, TDevice, CategoryTags::Batch, CategoryTags::ThreeDArray>;
}