#pragma once

#include <type_traits>
#include <MetaNN/facilities/traits.h>
#include <MetaNN/policies/policy_container.h>
#include <MetaNN/facilities/cont_metafuns/sequential.h>

namespace MetaNN
{
/// ============== policy select ============================
namespace NSPolicySelect
{
template <typename TPolicyCont>
struct PolicySelRes;

template <typename TCurPolicy, typename... TOtherPolicies>
struct PolicySelRes<PolicyContainer<TCurPolicy, TOtherPolicies...>> : TCurPolicy, TOtherPolicies...
{}; 

template <typename TMajorClass>
struct MajorFilter_
{
    template <typename TState, typename TInput>
    using apply = std::conditional_t<std::is_same_v<typename TInput::MajorClass, TMajorClass>,
                                     Sequential::PushBack_<TState, TInput>,
                                     Identity_<TState>
                                    >;
};

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
                                     Sequential::PushBack_<TState, TInput>>;
};
}

template <typename TSubPolicies, typename TParentPolicies>
using PolicyDerive = Sequential::Fold<TSubPolicies, TParentPolicies,
                                                   NSPolicyDerive::Filter_<TSubPolicies>::template apply>;
/// ============== plain policy ============================
namespace NSPlainPolicy
{
struct imp
{
    template <typename TState, typename TInput>
    struct apply
    {
        using type = Sequential::PushBack<TState, TInput>;
    };
    
    template <typename TState, typename TLayerName, typename... TAdded>
    struct apply<TState, SubPolicyContainer<TLayerName, TAdded...>>
    {
        using type = TState;
    };
};
}

template <typename TPolicyContainer>
using PlainPolicy = Sequential::Fold<PolicyContainer<>, TPolicyContainer,
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
    using tmp = Sequential::Fold<PolicyContainer<>, TPolicyContainer,
                                              imp_<TLayerName>::template apply>;
    using type = PolicyDerive<tmp, PlainPolicy<TPolicyContainer>>;
};
}

template <typename TPolicyContainer, typename TLayerName>
using SubPolicyPicker = typename NSSubPolicyPicker::SubPolicyPicker_<TPolicyContainer, TLayerName>::type;

/// ============== change policy ===============================
namespace NSChangePolicy
{
    template <typename TNewPolicy>
    struct Filter_
    {
        using TNewMajor = typename TNewPolicy::MajorClass;
        using TNewMinor = typename TNewPolicy::MinorClass;
        
        template <typename TState, typename TInput>
        struct apply : std::conditional_t<std::is_same_v<typename TInput::MajorClass, TNewMajor> &&
                                          std::is_same_v<typename TInput::MinorClass, TNewMinor>,
                                          Identity_<TState>,
                                          Sequential::PushBack_<TState, TInput>>
        {};
        
        template <typename TState, typename TLayer, typename...TParams>
        struct apply<TState, SubPolicyContainer<TLayer, TParams...>>
        {
            using type = Sequential::PushBack<TState, SubPolicyContainer<TLayer, TParams...>>;
        };
    };
}

template <typename TNewPolicy, typename TOriContainer>
struct ChangePolicy_
{
    using type = Sequential::PushBack<Sequential::Fold<PolicyContainer<>, TOriContainer,
                                                       NSChangePolicy::Filter_<TNewPolicy>::template apply>,
                                      TNewPolicy>;    
};

template <typename TNewPolicy, typename TOriContainer>
using ChangePolicy = typename ChangePolicy_<TNewPolicy, TOriContainer>::type;

/// ============== pick policy object ==========================
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
    using type =
        typename std::conditional_t<IsThePolicy,
                                    Identity_<TCurPolicy>,
                                    PickPolicyOjbect_<PolicyContainer<TPolicies...>, TMajorClass, TMinorClass>>::type;
};

template <typename TPolicyContainer, typename TMajorClass, typename TMinorClass>
using PickPolicyOjbect = typename PickPolicyOjbect_<TPolicyContainer, TMajorClass, TMinorClass>::type;

/// ============== has non-travil policy =======================
template <typename TPolicyContainer, typename TMajorClass, typename TMinorClass>
struct HasNonTrivialPolicy_;

template <typename TMajorClass, typename TMinorClass, typename... TPolicies>
struct HasNonTrivialPolicy_<PolicyContainer<TPolicies...>, TMajorClass, TMinorClass>
{
    constexpr static bool value = false;
};

template <typename TMajorClass, typename TMinorClass, typename TCurPolicy, typename... TPolicies>
struct HasNonTrivialPolicy_<PolicyContainer<TCurPolicy, TPolicies...>, TMajorClass, TMinorClass>
{
    constexpr static bool IsThePolicy = std::is_same_v<typename TCurPolicy::MajorClass, TMajorClass> &&
                                        std::is_same_v<typename TCurPolicy::MinorClass, TMinorClass>;
    constexpr static bool value = OrValue<IsThePolicy,
                                          HasNonTrivialPolicy_<PolicyContainer<TPolicies...>, TMajorClass, TMinorClass>>;
};

template <typename TPolicyContainer, typename TMajorClass, typename TMinorClass>
constexpr static bool HasNonTrivialPolicy = HasNonTrivialPolicy_<TPolicyContainer, TMajorClass, TMinorClass>::value;

}