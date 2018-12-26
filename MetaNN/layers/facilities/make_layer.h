#pragma once

#include <MetaNN/policies/policy_container.h>
namespace MetaNN
{
    template<template <typename, typename> class T, typename TInputMap, typename...TPolicies>
    using MakeLayer = T<TInputMap, PolicyContainer<TPolicies...>>;
}