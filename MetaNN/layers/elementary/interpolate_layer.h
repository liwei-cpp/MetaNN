#pragma once
#include <MetaNN/facilities/var_type_dict.h>
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/policies/policy_operations.h>

namespace MetaNN
{
using InterpolateLayerInput = VarTypeDict<struct InterpolateLayerWeight1,
                                          struct InterpolateLayerWeight2,
                                          struct InterpolateLayerLambda>;

template <typename TPolicies>
class InterpolateLayer
{
    static_assert(IsPolicyContainer<TPolicies>, "TPolicies is not a policy container.");
    using CurLayerPolicy = PlainPolicy<TPolicies>;

public:
    static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsFeedbackOutput;
    static constexpr bool IsUpdate = false;
    using InputType = InterpolateLayerInput;
    using OutputType = LayerIO;

public:
    template <typename TIn>
    auto FeedForward(const TIn& p_in)
    {
        const auto& val1 = p_in.template Get<InterpolateLayerWeight1>();
        const auto& val2 = p_in.template Get<InterpolateLayerWeight2>();
        const auto& val_lambda = p_in.template Get<InterpolateLayerLambda>();

        using rawType1 = std::decay_t<decltype(val1)>;
        using rawType2 = std::decay_t<decltype(val2)>;
        using rawType3 = std::decay_t<decltype(val_lambda)>;

        static_assert(!std::is_same<rawType1, NullParameter>::value, "parameter1 is invalid");
        static_assert(!std::is_same<rawType2, NullParameter>::value, "parameter2 is invalid");
        static_assert(!std::is_same<rawType3, NullParameter>::value, "parameter lambda is invalid");

        if constexpr(IsFeedbackOutput)
        {
            m_weight1.push(MakeDynamic(val1));
            m_weight2.push(MakeDynamic(val2));
            m_weight_lambda.push(MakeDynamic(val_lambda));
        }
        return LayerIO::Create().template Set<LayerIO>(Interpolate(val1, val2, val_lambda));
    }

    template <typename TGrad>
    auto FeedBackward(const TGrad& p_grad)
    {
        if constexpr (IsFeedbackOutput)
        {
            if ((m_weight1.empty()) || (m_weight2.empty()) || (m_weight_lambda.empty()))
            {
                throw std::runtime_error("Cannot do FeedBackward for InterpolateLayer");
            }
            auto tmp = p_grad.template Get<LayerIO>();
            auto res1 = m_weight_lambda.top() * tmp;
        
            auto res2 = (Scalar<int>(1) - m_weight_lambda.top()) * tmp;
            auto res_lambda = (m_weight1.top() - m_weight2.top()) * tmp;
            auto res = InterpolateLayerInput::Create().template Set<InterpolateLayerWeight1>(std::move(res1))
                                                .template Set<InterpolateLayerWeight2>(std::move(res2))
                                                .template Set<InterpolateLayerLambda>(std::move(res_lambda));
            m_weight_lambda.pop();
            m_weight1.pop();
            m_weight2.pop();
            return res;
        }
        else
        {
            return InterpolateLayerInput::Create();
        }
    }

    void NeutralInvariant()
    {
        if constexpr(IsFeedbackOutput)
        {
            if ((!m_weight1.empty()) || (!m_weight2.empty()) || (!m_weight_lambda.empty()))
            {
                throw std::runtime_error("NeutralInvariant Fail!");
            }
        }
    }

private:
    using DataType = LayerTraits::LayerInternalBuf<IsFeedbackOutput,
                                                   PolicySelect<InputPolicy, CurLayerPolicy>::BatchMode,
                                                   typename PolicySelect<OperandPolicy, CurLayerPolicy>::Element,
                                                   typename PolicySelect<OperandPolicy, CurLayerPolicy>::Device,
                                                   CategoryTags::Matrix, CategoryTags::BatchMatrix>;
    DataType m_weight1;
    DataType m_weight2;
    DataType m_weight_lambda;
};
}
