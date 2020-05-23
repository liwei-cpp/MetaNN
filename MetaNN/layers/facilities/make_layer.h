#pragma once

#include <MetaNN/policies/policy_container.h>
#include <MetaNN/layers/facilities/layer_in_map.h>

namespace MetaNN
{
    template<template <typename, typename> class T, typename...TPolicies>
    struct MakeInferLayer_
    {
        using type = T<NullParameter,
                       PolicyContainer<TPolicies...>>;
        static_assert(!type::IsFeedbackOutput);
        static_assert(!type::IsUpdate);
    };

    template<template <typename, typename> class T, typename...TPolicies>
    struct MakeInferLayer_<T, PolicyContainer<TPolicies...>> : MakeInferLayer_<T, TPolicies...> {};

    template<template <typename, typename> class T, typename...TPolicies>
    using MakeInferLayer = typename MakeInferLayer_<T, TPolicies...>::type;

    template<template <typename, typename> class T, typename TInputMap, typename...TPolicies>
    struct MakeTrainLayer_
    {
        using type = T<TInputMap, PolicyContainer<TPolicies...>>;
    };

    template<template <typename, typename> class T, typename TInputMap, typename...TPolicies>
    struct MakeTrainLayer_<T, TInputMap, PolicyContainer<TPolicies...>> : MakeTrainLayer_<T, TInputMap, TPolicies...> {};

    template<template <typename, typename> class T, typename TInputMap, typename...TPolicies>
    using MakeTrainLayer = typename MakeTrainLayer_<T, TInputMap, TPolicies...>::type;
}