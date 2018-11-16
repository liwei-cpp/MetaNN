#pragma once

#include <MetaNN/data/linear_table/linear_table_pack.h>

namespace MetaNN
{
template<typename TElem, typename TDevice, typename TCategory>
using Batch = StaticArray<TElem, TDevice, CategoryTags::Batch, TCategory>;
}