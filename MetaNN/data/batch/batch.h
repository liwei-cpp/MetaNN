#pragma once

#include <MetaNN/data/linear_table/linear_table_pack.h>

namespace MetaNN
{
template<typename TElem, typename TDevice, typename TCategory>
using Batch = LinearTable<TElem, TDevice, CategoryTags::Batch<TCategory>>;

template <typename TElement, typename TDevice, typename TCategory>
struct DataCategory_<Batch<TElement, TDevice, TCategory>>
{
    using type = CategoryTags::Batch<TCategory>;
};
}