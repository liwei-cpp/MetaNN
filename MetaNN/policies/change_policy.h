#pragma once

#include <MetaNN/policies/policy_container.h>
#include <MetaNN/facilities/cont_metafuns/sequential.h>

namespace MetaNN
{
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
}