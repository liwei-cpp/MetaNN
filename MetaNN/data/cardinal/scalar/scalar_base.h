#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>

namespace MetaNN
{
template <typename TElem, typename TDevice = DeviceTags::CPU>
class Scalar;

template <typename TElem, typename TDevice>
struct DataCategory_<Scalar<TElem, TDevice>>
{
    using type = CategoryTags::Scalar;
};
}