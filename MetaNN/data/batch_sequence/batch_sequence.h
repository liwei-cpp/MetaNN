#pragma once

#include <MetaNN/data/linear_table/linear_table_pack.h>

namespace MetaNN
{
template<typename TElem, typename TDevice>
using BatchScalarSequence = StaticArray<TElem, TDevice, CategoryTags::BatchSequence, CategoryTags::Scalar>;

template<typename TElem, typename TDevice>
using BatchMatrixSequence = StaticArray<TElem, TDevice, CategoryTags::BatchSequence, CategoryTags::Matrix>;

template<typename TElem, typename TDevice>
using BatchThreeDArraySequence = StaticArray<TElem, TDevice, CategoryTags::BatchSequence, CategoryTags::ThreeDArray>;
}