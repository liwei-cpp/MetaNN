#pragma once

#include <MetaNN/model_rel/param_initializer/facilities/policies.h>

namespace MetaNN
{
namespace NSParamInit
{
template <typename TPolicyCont, typename TGroup>
struct Group2Initializer_;

template <typename TPolicyCont>
struct Group2Initializer_<TPolicyCont, InitPolicy::WeightTypeCate>
{
    using type = typename PolicySelect<InitPolicy, TPolicyCont>::Weight;
};

template <typename TPolicyCont>
struct Group2Initializer_<TPolicyCont, InitPolicy::BiasTypeCate>
{
    using type = typename PolicySelect<InitPolicy, TPolicyCont>::Bias;
};

template <typename TPolicyCont, typename TSpecInit>
struct PickInitializerBySpec_
{
    using type = TSpecInit;
};

template <typename TPolicyCont>
struct PickInitializerBySpec_<TPolicyCont, void>
{
    using type = typename PolicySelect<InitPolicy, TPolicyCont>::Overall;
};

template <typename TPolicyCont, typename TSpecInitializer>
struct PickInitializer_
{
    static_assert(IsPolicyContainer<TPolicyCont>);
    using CurInitPolicy = PlainPolicy<TPolicyCont>;
    
    static_assert(!std::is_same<TSpecInitializer, InitPolicy::OverallTypeCate>::value);
    
    using SpecInitializer = typename Group2Initializer_<CurInitPolicy, TSpecInitializer>::type;
    
    using type = typename PickInitializerBySpec_<CurInitPolicy, SpecInitializer>::type;
};
}

template <typename TPolicyCont, typename TSpecInit>
using PickInitializer = typename NSParamInit::PickInitializer_<TPolicyCont, TSpecInit>::type;

namespace NSParamInit
{
template <typename TRes, typename...TPolicies>
struct FillerTagFromPolicy_
{
    using type = TRes;
};

template <typename...TRes, typename TCur, typename...TRem>
struct FillerTagFromPolicy_<std::tuple<TRes...>, PInitializerIs<TCur>, TRem...>
{
    using type = typename FillerTagFromPolicy_<std::tuple<TRes..., TCur>, TRem...>::type;
};

template <typename...TRes, typename TCur, typename...TRem>
struct FillerTagFromPolicy_<std::tuple<TRes...>, PWeightInitializerIs<TCur>, TRem...>
{
    using type = typename FillerTagFromPolicy_<std::tuple<TRes..., TCur>, TRem...>::type;
};

template <typename...TRes, typename TCur, typename...TRem>
struct FillerTagFromPolicy_<std::tuple<TRes...>, PBiasInitializerIs<TCur>, TRem...>
{
    using type = typename FillerTagFromPolicy_<std::tuple<TRes..., TCur>, TRem...>::type;
};

template <typename...TRes, typename TLayer, typename...TSub, typename...TRem>
struct FillerTagFromPolicy_<std::tuple<TRes...>, SubPolicyContainer<TLayer, TSub...>, TRem...>
{
    using step1 = typename FillerTagFromPolicy_<std::tuple<TRes...>, TSub...>::type;
    using type = typename FillerTagFromPolicy_<step1, TRem...>::type;
};

template <typename TChecker, typename...TCheckes>
struct AlreadyExist_
{
    constexpr static bool value = false;
};

template <typename TChecker, typename TCur, typename...TRem>
struct AlreadyExist_<TChecker, TCur, TRem...>
{
    constexpr static bool value = OrValue<std::is_same<TChecker, TCur>::value,
                                          AlreadyExist_<TChecker, TRem...>>;
};

template <typename TRes, typename TPolicyTuple>
struct TagDedupe_
{
    using type = TRes;
};

template <typename...TRes, typename TCur, typename...TRem>
struct TagDedupe_<VarTypeDict<TRes...>, std::tuple<TCur, TRem...>>
{
    using type = typename std::conditional_t<AlreadyExist_<TCur, TRes...>::value,
                                             TagDedupe_<VarTypeDict<TRes...>, std::tuple<TRem...>>,
                                             TagDedupe_<VarTypeDict<TRes..., TCur>, std::tuple<TRem...>>>::type;
};

template <typename...TPolicies>
struct FillerTags2NamedParams_
{
    using step1 = typename FillerTagFromPolicy_<std::tuple<>, TPolicies...>::type;
    using type = typename TagDedupe_<VarTypeDict<>, step1>::type;
};
}

template <typename...TPolicies>
using FillerTags2NamedParams = typename NSParamInit::FillerTags2NamedParams_<TPolicies...>::type;
}
