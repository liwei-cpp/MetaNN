#pragma once

#include <MetaNN/data/facilities/continuous_memory.h>
#include <MetaNN/data/facilities/lower_access.h>
#include <MetaNN/data/facilities/tags.h>

namespace MetaNN
{
template<typename TElement, typename TDevice, typename TCategory>
class Batch;

template <typename TElement, typename TDevice>
constexpr bool IsBatchMatrix<Batch<TElement, TDevice, CategoryTags::Matrix>> = true;

template <typename TElement, typename TDevice>
constexpr bool IsBatchScalar<Batch<TElement, TDevice, CategoryTags::Scalar>> = true;
}