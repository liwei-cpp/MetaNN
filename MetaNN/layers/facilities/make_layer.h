#pragma once

#include <MetaNN/policies/policy_container.h>
#include <MetaNN/layers/facilities/layer_io_map.h>

namespace MetaNN
{
    template<template <typename, typename, typename> class T, typename TInputMap, typename...TPolicies>
    using MakeLayer = T<TInputMap, LayerIOMap<>, PolicyContainer<TPolicies...>>;
    
    template<template <typename, typename, typename> class T, typename TInputMap, typename TGradMap, typename...TPolicies>
    using MakeBPLayer = T<TInputMap, TGradMap, PolicyContainer<TPolicies...>>;
}