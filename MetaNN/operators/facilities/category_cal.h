#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <tuple>

namespace MetaNN
{
template <typename TOpTag, typename THeadCate, typename...TRemainCate>
struct OperCategory_;

namespace NSOperCateCal
{
/// Same category check
template <typename TCate, typename...TRemain>
struct SameCate_
{
    constexpr static bool value = true;
};

template <typename TCate, typename TCur, typename...TRemain>
struct SameCate_<TCate, TCur, TRemain...>
{
    constexpr static bool tmp = std::is_same<TCate, TCur>::value;
    constexpr static bool value = AndValue<tmp,
                                           SameCate_<TCate, TRemain...>>;
};

template <typename TCateCont, typename...TData>
struct Data2Cate_
{
    using type = TCateCont;
};

template <typename...TProcessed, typename TCur, typename...TRemain>
struct Data2Cate_<std::tuple<TProcessed...>, TCur, TRemain...>
{
    using tmp1 = DataCategory<TCur>;
    using tmp2 = std::tuple<TProcessed..., tmp1>;
    using type = typename Data2Cate_<tmp2, TRemain...>::type;
};

template <typename THead, typename...TRemain>
using Data2Cate = typename Data2Cate_<std::tuple<>, THead, TRemain...>::type;

/// category sequence to category result
template <typename TOpTag, typename TCateContainer>
struct CateInduce_;

template <typename TOpTag, typename...TCates>
struct CateInduce_<TOpTag, std::tuple<TCates...>>
{
    using type = typename OperCategory_<TOpTag, TCates...>::type;
};
}

template <typename TOpTag, typename THeadCate, typename...TRemainCate>
struct OperCategory_
{
    static_assert(NSOperCateCal::SameCate_<THeadCate, TRemainCate...>::value,
                  "Data category mismatch.");
    using type = THeadCate;
};

template <typename TOpTag, typename THead, typename...TRemain>
using OperCateCal = typename NSOperCateCal::CateInduce_<TOpTag, 
                                                        NSOperCateCal::Data2Cate<THead, TRemain...>>::type;
}