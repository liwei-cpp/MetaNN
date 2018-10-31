#pragma once

#include <MetaNN/data/linear_table/linear_table_pack.h>

namespace MetaNN
{
template<typename TElem, typename TDevice, typename TCategory>
using Sequence = LinearTable<TElem, TDevice, CategoryTags::Sequence<TCategory>>;

template <typename TElement, typename TDevice, typename TCategory>
struct DataCategory_<Sequence<TElement, TDevice, TCategory>>
{
    using type = CategoryTags::Sequence<TCategory>;
};
}