#pragma once

#include <MetaNN/data/facilities/continuous_memory.h>
#include <MetaNN/data/facilities/lower_access.h>
#include <MetaNN/data/facilities/tags.h>

namespace MetaNN
{
template<typename TElement, typename TDevice, typename TCategory>
class Batch;

template <typename TElement, typename TDevice, typename TCategory>
struct DataCategory_<Batch<TElement, TDevice, TCategory>>
{
    using type = CategoryTags::Batch<TCategory>;
};
}