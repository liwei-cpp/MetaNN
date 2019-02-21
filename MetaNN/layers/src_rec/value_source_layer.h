#pragma once

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
        using ValueType = double;
        
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

    struct LayerOutput;

    template <typename TInputs, typename TGrads, typename TPolicies>
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
        using InputMap = LayerIOMap<>;
        using GradMap = LayerIOMap<LayerKV<LayerOutput, ValueType>>;
        
    public:
        ValueSourceLayer(std::string name)
            : m_name(std::move(name))
        {}
        
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            static const ValueType val = static_cast<ValueType>(Numerator * 1.0 / Denominator);
            return LayerOutputCont<ValueSourceLayer>().template Set<LayerOutput>(val);
        }

        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            return LayerInputCont<ValueSourceLayer>();
        }
    private:
        std::string m_name;
    };
}