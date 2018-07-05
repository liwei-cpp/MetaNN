#pragma once

#include <type_traits>
#include <MetaNN/facilities/traits.h>
#include <MetaNN/policies/policy_container.h>

namespace MetaNN
{
///  ============ policy exist check  ============================
template <typename TCont, typename TPolicy>
struct PolicyExist_;

template <typename T1, typename...T2, typename TPolicy>
struct PolicyExist_<PolicyContainer<T1, T2...>, TPolicy>
{
    using MJ1 = typename T1::MajorClass;
    using MJ2 = typename TPolicy::MajorClass;
    using MI1 = typename T1::MinorClass;
    using MI2 = typename TPolicy::MinorClass;

    constexpr static bool tmp1 = std::is_same<MJ1, MJ2>::value;
    constexpr static bool tmp2 = AndValue<tmp1, std::is_same<MI1, MI2>>;
    constexpr static bool value = OrValue<tmp2, PolicyExist_<PolicyContainer<T2...>, TPolicy>>;
};

template <typename TLayerName, typename...T1, typename...T2, typename TPolicy>
struct PolicyExist_<PolicyContainer<SubPolicyContainer<TLayerName, T1...>, T2...>, TPolicy>
{
    constexpr static bool value = PolicyExist_<PolicyContainer<T2...>, TPolicy>::value;
};

template <typename TPolicy>
struct PolicyExist_<PolicyContainer<>, TPolicy>
{
    constexpr static bool value = false;
};

template <typename TArray, typename TPolicy>
constexpr static bool PolicyExist = PolicyExist_<TArray, TPolicy>::value;

/// ============== policy derive ============================
namespace NSPolicyDerive
{
template <typename TProcessingCont, typename TFiltedCont, typename TCompPoliCont>
struct PolicyDeriveFil_;

template <typename TCompPoliCont, typename TCur, typename...TProcessings, typename...TProcessed>
struct PolicyDeriveFil_<PolicyContainer<TCur, TProcessings...>,
                        PolicyContainer<TProcessed...>,
                        TCompPoliCont>
{
    constexpr static bool dupe = PolicyExist<TCompPoliCont, TCur>;
    using TNewFiltered = std::conditional_t<dupe,
                                            PolicyContainer<TProcessed...>,
                                            PolicyContainer<TProcessed..., TCur>>;
    using type = typename PolicyDeriveFil_<PolicyContainer<TProcessings...>,
                                           TNewFiltered,
                                           TCompPoliCont>::type;
};

template <typename TCompPoliCont, typename...TProcessed>
struct PolicyDeriveFil_<PolicyContainer<>,
                        PolicyContainer<TProcessed...>,
                        TCompPoliCont>
{
    using type = PolicyContainer<TProcessed...>;
};

template <typename TModCont, typename TAddCont>
struct PolicyCascade_;

template <typename...TMod, typename...TAdd>
struct PolicyCascade_<PolicyContainer<TMod...>, PolicyContainer<TAdd...>>
{
    using type = PolicyContainer<TMod..., TAdd...>;
};

template <typename TSubPolicies, typename TParentPolicies>
struct PolicyDerive_
{
    using TFiltered = typename PolicyDeriveFil_<TParentPolicies,
                                                PolicyContainer<>,
                                                TSubPolicies>::type;
    using type = typename PolicyCascade_<TSubPolicies, TFiltered>::type;
};
}

template <typename TSubPolicies, typename TParentPolicies>
using PolicyDerive = typename NSPolicyDerive::PolicyDerive_<TSubPolicies, TParentPolicies>::type;

/// ============== plain policy ============================
template <typename TPolicyContainer, typename TResContainer>
struct PlainPolicy_
{
    using type = TResContainer;
};

template <typename TCurPolicy, typename...TPolicies, typename... TFilteredPolicies>
struct PlainPolicy_<PolicyContainer<TCurPolicy, TPolicies...>,
                    PolicyContainer<TFilteredPolicies...>>
{
    using TNewFiltered = std::conditional_t<IsSubPolicyContainer<TCurPolicy>,
                                            PlainPolicy_<PolicyContainer<TPolicies...>,
                                                         PolicyContainer<TFilteredPolicies...>>,
                                            PlainPolicy_<PolicyContainer<TPolicies...>,
                                                         PolicyContainer<TFilteredPolicies..., TCurPolicy>>>;
    using type = typename TNewFiltered::type;
};

template <typename TPolicyContainer>
using PlainPolicy = typename PlainPolicy_<TPolicyContainer, PolicyContainer<>>::type;

/// ============== sub policy picker============================
namespace NSSubPolicyPicker
{
template <typename TPolicyContainer, typename TLayerName>
struct SPP_
{
    using type = PolicyContainer<>;
};

template <typename TCur, typename...TPolicies, typename TLayerName>
struct SPP_<PolicyContainer<TCur, TPolicies...>, TLayerName>
{
    using type = typename SPP_<PolicyContainer<TPolicies...>, TLayerName>::type;
};

template <typename...TCur, typename...TPolicies, typename TLayerName>
struct SPP_<PolicyContainer<SubPolicyContainer<TLayerName, TCur...>, TPolicies...>, TLayerName>
{
    using type = PolicyContainer<TCur...>;
};

template <typename TPolicyContainer, typename TLayerName>
struct SubPolicyPicker_
{
    using tmp1 = typename SPP_<TPolicyContainer, TLayerName>::type;
    using tmp2 = PlainPolicy<TPolicyContainer>;
    using type = PolicyDerive<tmp1, tmp2>;
};
}

template <typename TPolicyContainer, typename TLayerName>
using SubPolicyPicker = typename NSSubPolicyPicker::SubPolicyPicker_<TPolicyContainer, TLayerName>::type;
}