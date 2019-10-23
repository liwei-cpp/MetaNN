#pragma once

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace MetaNN
{
template <typename T>
struct Identity_
{
    using type = T;
};

// Or
template <bool cur, typename TNext>
constexpr static bool OrValue = true;

template <typename TNext>
constexpr static bool OrValue<false, TNext> = TNext::value;

// And
template <bool cur, typename TNext>
constexpr static bool AndValue = false;

template <typename TNext>
constexpr static bool AndValue<true, TNext> = TNext::value;

template <typename T>
using RemConstRef = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename... T>
constexpr static bool DependencyFalse = false;
}