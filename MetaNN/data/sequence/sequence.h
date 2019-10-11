#pragma once

#include <MetaNN/data/linear_table/_.h>

namespace MetaNN
{
template<typename TElem, typename TDevice>
using ScalarSequence = StaticArray<TElem, TDevice, CategoryTags::Sequence, CategoryTags::Scalar>;

template<typename TElem, typename TDevice>
using MatrixSequence = StaticArray<TElem, TDevice, CategoryTags::Sequence, CategoryTags::Matrix>;

template<typename TElem, typename TDevice>
using ThreeDArraySequence = StaticArray<TElem, TDevice, CategoryTags::Sequence, CategoryTags::ThreeDArray>;

template <typename TData>
using DynamicSequence = DynamicArray<TData, CategoryTags::Sequence>;

template <typename TIterator>
auto MakeDynamicSequence(TIterator beg, TIterator end)
{
    using TData = typename std::iterator_traits<TIterator>::value_type;
    using RawData = RemConstRef<TData>;
    return DynamicSequence<RawData>(beg, end);
}
}