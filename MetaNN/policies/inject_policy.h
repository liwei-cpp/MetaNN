#pragma once

#include <MetaNN/policies/policy_container.h>
namespace MetaNN
{
//template <typename...TParams>
//template<template <typename TPolicyCont, typename...> class T, typename...TPolicies>
//using InjectPolicy = T<PolicyContainer<TPolicies...>, TParams...>;

template<template <typename, typename> class T, typename TInputMap, typename...TPolicies>
using MakeLayer = T<TInputMap, PolicyContainer<TPolicies...>>;
}