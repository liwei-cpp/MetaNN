#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <tuple>

namespace MetaNN
{
template <typename TOpTag, typename THeadCate, typename...TRemainCate>
struct OperCategory_
{
    static_assert((std::is_same_v<THeadCate, TRemainCate> && ...), "Data category mismatch.");
    using type = THeadCate;
};

template <typename TOpTag, typename THead, typename...TRemain>
using OperCateCal = typename OperCategory_<TOpTag, DataCategory<THead>,
                                           DataCategory<TRemain>...>::type;
}