#pragma once

#include <MetaNN/policies/policy_container.h>
#include <MetaNN/layers/facilities/layer_io_map.h>

namespace MetaNN
{
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
	
	
	namespace NSMakeTrainLayer
	{
		template <typename TInputMap, typename TKeySet>
		struct CheckInputMapAvailable_;
		
		template <typename... TKVs, typename TKeySet>
		struct CheckInputMapAvailable_<LayerIOMap<TKVs...>, TKeySet>
		{
			constexpr static bool value = (ContMetaFun::Set::HasKey<TKeySet, typename TKVs::KeyType> && ...);
		};
	}
	
    template<template <typename, typename, typename> class T, typename TInputMap, typename...TPolicies>
    struct MakeTrainLayer_
    {
		using InputKeys = typename LayerInputPortSet_<T>::type;
		static_assert(NSMakeTrainLayer::CheckInputMapAvailable_<TInputMap, InputKeys>::value);
		static_assert(ArraySize<TInputMap> == ArraySize<InputKeys>);
        using type = T<TInputMap,
                       EmptyLayerIOMap<typename LayerOutputPortSet_<T>::type>,
                       PolicyContainer<TPolicies...>>;
    };
	
    template<template <typename, typename, typename> class T, typename TInputMap, typename...TPolicies>
    struct MakeTrainLayer_<T, TInputMap, PolicyContainer<TPolicies...>> : MakeTrainLayer_<T, TInputMap, TPolicies...> {};
	
	template<template <typename, typename, typename> class T, typename TInputMap, typename...TPolicies>
    using MakeTrainLayer = typename MakeTrainLayer_<T, TInputMap, TPolicies...>::type;
}