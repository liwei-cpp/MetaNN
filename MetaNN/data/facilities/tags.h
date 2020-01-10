#pragma once

#include <type_traits>

namespace MetaNN
{
/// data types
namespace CategoryTags
{
    struct OutOfCategory;
    
    struct Scalar;
    struct Tensor;
}

template <typename T>
constexpr bool IsValidCategoryTag = std::is_same_v<T, CategoryTags::Scalar> ||
                                    std::is_same_v<T, CategoryTags::Tensor>;

/// device types
struct DeviceTags
{
    struct CPU;
};
}
