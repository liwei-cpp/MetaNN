#pragma once

#include <type_traits>
#include <MetaNN/facilities/traits.h>
#include <MetaNN/policies/policy_container.h>
#include <MetaNN/facilities/cont_metafuns/sequential.h>

namespace MetaNN
{
namespace NSPolicySelect
{
template <typename TPolicyCont>
struct PolicySelRes;

template <typename TCurPolicy, typename... TOtherPolicies>
struct PolicySelRes<PolicyContainer<TCurPolicy, TOtherPolicies...>> : TCurPolicy, TOtherPolicies...
{}; 

/// ================= Major Filter =============================
template <typename TMajorClass>
struct MajorFilter_
{
    template <typename TState, typename TInput>
    using apply = typename std::conditional_t<std::is_same_v<typename TInput::MajorClass, TMajorClass>,
                                              Sequential::PushBack_<TState, TInput>,
                                              Identity_<TState>
                                             >;
};

/// ================= Minor Check ===============================
template <typename TPolicyCont>
struct MinorCheck_
{
    static constexpr bool value = true;
};

template <typename TCurPolicy, typename... TP>
struct MinorCheck_<PolicyContainer<TCurPolicy, TP...>>
{
    static constexpr bool cur_check = ((!std::is_same_v<typename TCurPolicy::MinorClass,
                                                        typename TP::MinorClass>) && ...);
                                                        
    static constexpr bool value
        = AndValue<cur_check, MinorCheck_<PolicyContainer<TP...>>>;
};

template <typename TMajorClass, typename TPolicyContainer>
struct Selector_
{
    using TMF = Sequential::Fold<PolicyContainer<>, TPolicyContainer,
                                              MajorFilter_<TMajorClass>::template apply>;
    static_assert(MinorCheck_<TMF>::value, "Minor class set conflict!");
    using type = std::conditional_t<Sequential::Size<TMF> == 0, TMajorClass, PolicySelRes<TMF>>;
};
}

template <typename TMajorClass, typename TPolicyContainer>
using PolicySelect = typename NSPolicySelect::Selector_<TMajorClass, TPolicyContainer>::type;

template <typename TPolicyContainer, typename TMajorClass, typename TMinorClass>
struct PickPolicyOjbect_;

template <typename TMajorClass, typename TMinorClass, typename... TPolicies>
struct PickPolicyOjbect_<PolicyContainer<TPolicies...>, TMajorClass, TMinorClass>
{
    using type = TMajorClass;
};

template <typename TMajorClass, typename TMinorClass, typename TCurPolicy, typename... TPolicies>
struct PickPolicyOjbect_<PolicyContainer<TCurPolicy, TPolicies...>, TMajorClass, TMinorClass>
{
    constexpr static bool IsThePolicy = std::is_same_v<typename TCurPolicy::MajorClass, TMajorClass> &&
                                        std::is_same_v<typename TCurPolicy::MinorClass, TMinorClass>;
    using type = std::conditional_t<IsThePolicy,
                                    TCurPolicy,
                                    PickPolicyOjbect_<PolicyContainer<TPolicies...>, TMajorClass, TMinorClass>>;
};

template <typename TPolicyContainer, typename TMajorClass, typename TMinorClass>
using PickPolicyOjbect = typename PickPolicyOjbect_<TPolicyContainer, TMajorClass, TMinorClass>::type;

template <typename TPolicyContainer, typename TMajorClass, typename TMinorClass>
struct HasNonTrivalPolicy_;

template <typename TMajorClass, typename TMinorClass, typename... TPolicies>
struct HasNonTrivalPolicy_<PolicyContainer<TPolicies...>, TMajorClass, TMinorClass>
{
    constexpr static bool value = false;
};

template <typename TMajorClass, typename TMinorClass, typename TCurPolicy, typename... TPolicies>
struct HasNonTrivalPolicy_<PolicyContainer<TCurPolicy, TPolicies...>, TMajorClass, TMinorClass>
{
    constexpr static bool IsThePolicy = std::is_same_v<typename TCurPolicy::MajorClass, TMajorClass> &&
                                        std::is_same_v<typename TCurPolicy::MinorClass, TMinorClass>;
    constexpr static bool value = OrValue<IsThePolicy,
                                          HasNonTrivalPolicy_<PolicyContainer<TPolicies...>, TMajorClass, TMinorClass>>;
};

template <typename TPolicyContainer, typename TMajorClass, typename TMinorClass>
constexpr static bool HasNonTrivalPolicy = HasNonTrivalPolicy_<TPolicyContainer, TMajorClass, TMinorClass>::value;
}