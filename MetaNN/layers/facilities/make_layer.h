#pragma once

#include <MetaNN/policies/policy_container.h>
#include <MetaNN/layers/facilities/layer_io_map.h>

namespace MetaNN
{
    namespace NSMakeLayer
    {
        template<template <typename, typename, typename> class T, typename TInputMap, typename...TPolicies>
        struct MakeLayer_
        {
            using type = T<TInputMap, EmptyLayerIOMap<typename LayerOutputPortSet_<T>::type>, PolicyContainer<TPolicies...>>;
        };
        
        template<template <typename, typename, typename> class T, typename TInputMap, typename...TPolicies>
        struct MakeLayer_<T, TInputMap, PolicyContainer<TPolicies...>>
        {
            using type = T<TInputMap, EmptyLayerIOMap<typename LayerOutputPortSet_<T>::type>, PolicyContainer<TPolicies...>>;
        };
    }
    
    template<template <typename, typename, typename> class T, typename TInputMap, typename...TPolicies>
    using MakeLayer = typename NSMakeLayer::MakeLayer_<T, TInputMap, TPolicies...>::type;
    
    template<template <typename, typename, typename> class T, typename TInputMap, typename TGradMap, typename...TPolicies>
    using MakeBPLayer = T<TInputMap, TGradMap, PolicyContainer<TPolicies...>>;
    
    template<template <typename, typename, typename> class T, typename...TPolicies>
    struct MakeInferLayer_
    {
        using type = T<EmptyLayerIOMap<typename LayerInputPortSet_<T>::type>,
                       EmptyLayerIOMap<typename LayerOutputPortSet_<T>::type>,
                       PolicyContainer<TPolicies...>>;
        static_assert(!type::IsFeedbackOutput);
        static_assert(!type::IsUpdate);
    };
        
    template<template <typename, typename, typename> class T, typename...TPolicies>
    struct MakeInferLayer_<T, PolicyContainer<TPolicies...>> : MakeInferLayer_<T, TPolicies...> {};

    template<template <typename, typename, typename> class T, typename...TPolicies>
    using MakeInferLayer = typename MakeInferLayer_<T, TPolicies...>::type;
}