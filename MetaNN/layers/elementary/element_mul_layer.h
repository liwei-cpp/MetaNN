#pragma once

#include <MetaNN/facilities/var_type_dict.h>
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/policies/policy_operations.h>
#include <stdexcept>

namespace MetaNN
{
using ElementMulLayerInput = VarTypeDict<struct ElementMulLayerIn1,
                                         struct ElementMulLayerIn2>;

template <typename TPolicies>
class ElementMulLayer
{
    static_assert(IsPolicyContainer<TPolicies>, "TPolicies is not a policy container.");
    using CurLayerPolicy = PlainPolicy<TPolicies>;
public:
    static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsFeedbackOutput;
    static constexpr bool IsUpdate = false;
    using InputType = ElementMulLayerInput;
    using OutputType = LayerIO;

public:
    template <typename TIn>
    auto FeedForward(const TIn& p_in)
    {
        const auto& val1 = p_in.template Get<ElementMulLayerIn1>();
        const auto& val2 = p_in.template Get<ElementMulLayerIn2>();

        using rawType1 = std::decay_t<decltype(val1)>;
        using rawType2 = std::decay_t<decltype(val2)>;

        static_assert(!std::is_same<rawType1, NullParameter>::value, "parameter1 is invalid");
        static_assert(!std::is_same<rawType2, NullParameter>::value, "parameter2 is invalid");

        if constexpr (IsFeedbackOutput)
        {
            m_data1.push(MakeDynamic(val1));
            m_data2.push(MakeDynamic(val2));
        }
        return LayerIO::Create().template Set<LayerIO>(val1 * val2);
    }

    template <typename TGrad>
    auto FeedBackward(const TGrad& p_grad)
    {
        if constexpr (IsFeedbackOutput)
        {
            if ((m_data1.empty()) || (m_data2.empty()))
            {
                throw std::runtime_error("Cannot do FeedBackward for ElementMulLayer.");
            }

            auto top1 = m_data1.top();
            auto top2 = m_data2.top();
            m_data1.pop();
            m_data2.pop();

            auto grad_eval = p_grad.template Get<LayerIO>();

            return ElementMulLayerInput::Create()
                            .template Set<ElementMulLayerIn1>(grad_eval * top2)
                            .template Set<ElementMulLayerIn2>(grad_eval * top1);
        }
        else
        {
            return ElementMulLayerInput::Create();
        }
    }

    void NeutralInvariant()
    {
        if constexpr(IsFeedbackOutput)
        {
            if ((!m_data1.empty()) || (!m_data2.empty()))
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
    DataType m_data1;
    DataType m_data2;
};
}
