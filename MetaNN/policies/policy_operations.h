#pragma once

#include <type_traits>
#include <MetaNN/facilities/traits.h>
#include <MetaNN/policies/policy_container.h>
#include <MetaNN/facilities/cont_metafuns/sequential.h>

namespace MetaNN
{
/// ============== policy derive ============================
namespace NSPolicyDerive
{
template <typename TPolicy>
struct PolicyExistKV
{
    static std::true_type apply(typename TPolicy::MajorClass*, typename TPolicy::MinorClass*);
};

template <typename TLayer, typename... TParams>
struct PolicyExistKV<SubPolicyContainer<TLayer, TParams...>>
{
    // Just a placeholder.
    static void apply(SubPolicyContainer<TLayer, TParams...>*);
};

template <typename TSubPlicies>
struct PolicyExist;

template <typename... TPolicies>
struct PolicyExist<PolicyContainer<TPolicies...>> : PolicyExistKV<TPolicies> ...
{
    static std::false_type apply(...);
    using PolicyExistKV<TPolicies>::apply ... ;
};

template <typename TSubPolicies>
struct Filter_
{
    template <typename TState, typename TInput>
    using apply = std::conditional_t<decltype(PolicyExist<TSubPolicies>::apply((typename TInput::MajorClass*) nullptr,
                                                                               (typename TInput::MinorClass*) nullptr))::value,
                                     Identity_<TState>,
                                     ContMetaFun::Sequential::PushBack_<TState, TInput>>;
};
}

template <typename TSubPolicies, typename TParentPolicies>
using PolicyDerive = ContMetaFun::Sequential::Fold<TSubPolicies, TParentPolicies,
                                                   NSPolicyDerive::Filter_<TSubPolicies>::template apply>;
/// ============== plain policy ============================
namespace NSPlainPolicy
{
struct imp
{
    template <typename TState, typename TInput>
    struct apply
    {
        using type = ContMetaFun::Sequential::PushBack<TState, TInput>;
    };
    
    template <typename TState, typename TLayerName, typename... TAdded>
    struct apply<TState, SubPolicyContainer<TLayerName, TAdded...>>
    {
        using type = TState;
    };
};
}

template <typename TPolicyContainer>
using PlainPolicy = ContMetaFun::Sequential::Fold<PolicyContainer<>, TPolicyContainer,
                                                  NSPlainPolicy::imp::template apply>;

/// ============== sub policy picker============================
namespace NSSubPolicyPicker
{
template <typename TLayerName>
struct imp_
{
    template <typename TState, typename TInput>
    struct apply
    {
        using type = TState;
    };
    
    template <typename...TProcessed, typename... TAdded>
    struct apply<PolicyContainer<TProcessed...>, SubPolicyContainer<TLayerName, TAdded...>>
    {
        using type = PolicyContainer<TProcessed..., TAdded...>;
    };
};

template <typename TPolicyContainer, typename TLayerName>
struct SubPolicyPicker_
{
    using tmp = ContMetaFun::Sequential::Fold<PolicyContainer<>, TPolicyContainer,
                                              imp_<TLayerName>::template apply>;
    using type = PolicyDerive<tmp, PlainPolicy<TPolicyContainer>>;
};
}

template <typename TPolicyContainer, typename TLayerName>
using SubPolicyPicker = typename NSSubPolicyPicker::SubPolicyPicker_<TPolicyContainer, TLayerName>::type;
}