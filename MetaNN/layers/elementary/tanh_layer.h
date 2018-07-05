#pragma once
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/policies/policy_operations.h>

namespace MetaNN
{
template <typename TPolicies>
class TanhLayer
{
    static_assert(IsPolicyContainer<TPolicies>, "TPolicies is not a policy container.");
    using CurLayerPolicy = PlainPolicy<TPolicies>;

public:
    static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsFeedbackOutput;
    static constexpr bool IsUpdate = false;
    using InputType = LayerIO;
    using OutputType = LayerIO;

public:
    template <typename TIn>
    auto FeedForward(const TIn& p_in)
    {
        const auto& val = p_in.template Get<LayerIO>();

        using rawType = std::decay_t<decltype(val)>;
        static_assert(!std::is_same<rawType, NullParameter>::value, "parameter is invalid");

        auto tmp = Tanh(val);
        if constexpr (IsFeedbackOutput)
        {
            m_data.push(MakeDynamic(tmp));
        }
        return LayerIO::Create().template Set<LayerIO>(std::move(tmp));
    }

    template <typename TGrad>
    auto FeedBackward(const TGrad& p_grad)
    {
        if constexpr (IsFeedbackOutput)
        {
            if (m_data.empty())
            {
                throw std::runtime_error("Cannot feed back in SigmoidLayer");
            }
            auto tmp = p_grad.template Get<LayerIO>();
            auto& tmp2 = m_data.top();
            auto res = LayerIO::Create().template Set<LayerIO>(TanhDerivative(tmp, tmp2));
            m_data.pop();
            return res;
        }
        else
        {
            return LayerIO::Create();
        }
    }

    void NeutralInvariant()
    {
        if constexpr(IsFeedbackOutput)
        {
            if (!m_data.empty())
            {
                throw std::runtime_error("NeutralInvariant Fail!");
            }
        }
    }

private:
    using InternalType = LayerTraits::LayerInternalBuf<IsFeedbackOutput,
                                                       PolicySelect<InputPolicy, CurLayerPolicy>::BatchMode,
                                                       typename PolicySelect<OperandPolicy, CurLayerPolicy>::Element,
                                                       typename PolicySelect<OperandPolicy, CurLayerPolicy>::Device,
                                                       CategoryTags::Matrix, CategoryTags::BatchMatrix>;
    InternalType m_data;
};
}
