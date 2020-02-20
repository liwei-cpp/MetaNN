#pragma once

#include <MetaNN/facilities/var_type_dict.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>

namespace MetaNN
{
    struct ValueSourcePolicy
    {
        using MajorClass = ValueSourcePolicy;

        struct ValueTypeTypeCate;
        using ValueType = float;

    };
#include <MetaNN/policies/policy_macro_begin.h>
    TypePolicyTemplate(PValueTypeIs, ValueSourcePolicy, ValueType);
#include <MetaNN/policies/policy_macro_end.h>

    template <typename TInputs, typename TPolicies>
    class ValueSourceLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = false;
        static constexpr bool IsUpdate = false;

    private:
        using ValueType = typename PolicySelect<ValueSourcePolicy, CurLayerPolicy>::ValueType;
        constexpr static int Numerator = PolicySelect<ValueSourcePolicy, CurLayerPolicy>::Numerator;
        constexpr static int Denominator = PolicySelect<ValueSourcePolicy, CurLayerPolicy>::Denominator;
        static_assert(std::is_same_v<ValueType, RemConstRef<ValueType>>);

    public:
        using InputPortSet = LayerPortSet<>;
        using OutputPortSet = LayerPortSet<struct LayerOutput>;
        using InputMap = typename EmptyLayerIOMap_<InputPortSet>::type;
        
    public:
        ValueSourceLayer(std::string name, ValueType p_value)
            : m_name(std::move(name))
            , m_value(p_value)
        {}
        
        auto FeedForward(const VarTypeDict<>::Values<>&)
        {
            return LayerOutputCont<ValueSourceLayer>().template Set<LayerOutput>(m_value);
        }

        template <typename TGrad>
        auto FeedBackward(TGrad&&)
        {
            return LayerInputCont<ValueSourceLayer>();
        }
    private:
        std::string m_name;
        ValueType m_value;
    };
}