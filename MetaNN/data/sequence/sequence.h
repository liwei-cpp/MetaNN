#pragma once

#include <MetaNN/data/linear_table/linear_table_pack.h>

namespace MetaNN
{
template<typename TElem, typename TDevice>
using ScalarSequence = StaticArray<TElem, TDevice, CategoryTags::Sequence, CategoryTags::Scalar>;

template<typename TElem, typename TDevice>
using MatrixSequence = StaticArray<TElem, TDevice, CategoryTags::Sequence, CategoryTags::Matrix>;

template<typename TElem, typename TDevice>
using ThreeDArraySequence = StaticArray<TElem, TDevice, CategoryTags::Sequence, CategoryTags::ThreeDArray>;
}