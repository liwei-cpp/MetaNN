#pragma once

#include <MetaNN/data/linear_table/_.h>

namespace MetaNN
{
template<typename TElem, typename TDevice>
using BatchScalar = StaticArray<TElem, TDevice, CategoryTags::Batch, CategoryTags::Scalar>;

template<typename TElem, typename TDevice>
using BatchMatrix = StaticArray<TElem, TDevice, CategoryTags::Batch, CategoryTags::Matrix>;

template<typename TElem, typename TDevice>
using BatchThreeDArray = StaticArray<TElem, TDevice, CategoryTags::Batch, CategoryTags::ThreeDArray>;

template <typename TData>
using DynamicBatch = DynamicArray<TData, CategoryTags::Batch>;

template <typename TIterator>
auto MakeDynamicBatch(TIterator beg, TIterator end)
{
    using TData = typename std::iterator_traits<TIterator>::value_type;
    using RawData = RemConstRef<TData>;
    return DynamicBatch<RawData>(beg, end);
}
}