#pragma once

#include <MetaNN/policies/policy_container.h>

namespace MetaNN
{
template <typename TNewPolicy, typename TOriContainer>
struct ChangePolicy_;

template <typename TNewPolicy, typename... TPolicies>
struct ChangePolicy_<TNewPolicy, PolicyContainer<TPolicies...>>
{
private:
    using newMajor = typename TNewPolicy::MajorClass;
    using newMinor = typename TNewPolicy::MinorClass;

    template <typename TPC, typename... TP> struct DropAppend_;

    template <typename... TFilteredPolicies>
    struct DropAppend_<PolicyContainer<TFilteredPolicies...>>
    {
        using type = PolicyContainer<TFilteredPolicies..., TNewPolicy>;
    };

    template <typename... TFilteredPolicies, typename TCurPolicy, typename... TP>
    struct DropAppend_<PolicyContainer<TFilteredPolicies...>,
                       TCurPolicy, TP...>
    {
        template <bool isArray, typename TDummy = void>
        struct ArrayBasedSwitch_
        {
            template <typename TMajor, typename TMinor, typename TD = void>
            struct _impl
            {
                using type = PolicyContainer<TFilteredPolicies..., TCurPolicy>;
            };

            template <typename TD>
            struct _impl<newMajor, newMinor, TD>
            {
                using type = PolicyContainer<TFilteredPolicies...>;
            };
            using type = typename _impl<typename TCurPolicy::MajorClass,
                                        typename TCurPolicy::MinorClass>::type;
        };

        template <typename TDummy>
        struct ArrayBasedSwitch_<true, TDummy>
        {
            using type = PolicyContainer<TFilteredPolicies..., TCurPolicy>;
        };

        using t1 = typename ArrayBasedSwitch_<IsSubPolicyContainer<TCurPolicy>>::type;
        using type = typename DropAppend_<t1, TP...>::type;
    };

public:
    using type = typename DropAppend_<PolicyContainer<>, TPolicies...>::type;
};

template <typename TNewPolicy, typename TOriContainer>
using ChangePolicy = typename ChangePolicy_<TNewPolicy, TOriContainer>::type;
}