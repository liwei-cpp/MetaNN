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
        
        struct NumeratorValueCate;
        static constexpr int Numerator = 0;
        
        struct DenominatorValueCate;
        static constexpr int Denominator = 1;

    };
#include <MetaNN/policies/policy_macro_begin.h>
    TypePolicyTemplate(PValueTypeIs, ValueSourcePolicy, ValueType);
    ValuePolicyTemplate(PNumeratorIs, ValueSourcePolicy, Numerator);
    ValuePolicyTemplate(PDenominatorIs, ValueSourcePolicy, Denominator);
#include <MetaNN/policies/policy_macro_end.h>

    template <typename TInputs, typename TPolicies>
    class ValueSourceLayer;
    
    template <>
    struct LayerInputPortSet_<ValueSourceLayer<void, void>>
    {
        using type = LayerPortSet<>;
    };
    
    template <>
    struct LayerOutputPortSet_<ValueSourceLayer<void, void>>
    {
        using type = LayerPortSet<struct LayerOutput>;
    };

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

    public:
        using InputPortSet = LayerInputPortSet<ValueSourceLayer>;
        using OutputPortSet = LayerOutputPortSet<ValueSourceLayer>;
        using InputMap = TInputs;
        
    public:
        ValueSourceLayer(std::string name)
            : m_name(std::move(name))
        {}
        
        auto FeedForward(const VarTypeDict<>::Values<>&)
        {
            static const ValueType val = static_cast<ValueType>(Numerator * 1.0 / Denominator);
            return LayerOutputCont<ValueSourceLayer>().template Set<LayerOutput>(val);
        }

        template <typename TGrad>
        auto FeedBackward(TGrad&&)
        {
            return LayerInputCont<ValueSourceLayer>();
        }
    private:
        std::string m_name;
    };
}