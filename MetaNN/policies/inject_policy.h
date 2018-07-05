#pragma once

#include <MetaNN/policies/policy_container.h>
namespace MetaNN
{
//template <typename...TParams>
//template<template <typename TPolicyCont, typename...> class T, typename...TPolicies>
//using InjectPolicy = T<PolicyContainer<TPolicies...>, TParams...>;

template<template <typename TPolicyCont> class T, typename...TPolicies>
using InjectPolicy = T<PolicyContainer<TPolicies...>>;
}