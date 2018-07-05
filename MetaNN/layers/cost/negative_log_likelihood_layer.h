#pragma once
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/policies/policy_operations.h>

namespace MetaNN
{
namespace NSNegativeLogLikelihoodLayer
{
template <bool isFeedback>
struct Feedback_
{
    template <typename TLabel, typename TIn, typename TData>
    static void Record(const TLabel& label, const TIn& in,
                       TData& label_stack, TData& pred_stack)
    {
        label_stack.push(MakeDynamic(label));
        pred_stack.push(MakeDynamic(in));
    }

    template <typename TGrad, typename TData>
    static auto Feedback(const TGrad& p_grad, TData& label, TData& pred)
    {
        if ((label.empty()) || (pred.empty()))
        {
            throw std::runtime_error("Cannot do FeedBackward for Negative Log-likelihood Layer");
        }
        auto l = label.top();
        auto p = pred.top();
        label.pop();
        pred.pop();

        auto g = p_grad.template Get<LayerIO>();
        auto res = NegativeLogLikelihoodDerivative(g, std::move(l), std::move(p));
        return CostLayerIn::Create().template Set<CostLayerIn>(std::move(res));
    }
};

template <>
struct Feedback_<false>
{
    template <typename TLabel, typename TIn, typename TData>
    static void Record(TLabel&&, TIn&& p_in, TData&&, TData&&) { }

    template <typename TGrad, typename TData>
    static auto Feedback(TGrad&&, TData&&, TData&&)
    {
        return CostLayerIn::Create();
    }
};
}

template <typename TPolicies>
class NegativeLogLikelihoodLayer
{
    static_assert(IsPolicyContainer<TPolicies>, "TPolicies is not a policy container.");
    using CurLayerPolicy = PlainPolicy<TPolicies>;

public:
    static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsFeedbackOutput;
    static constexpr bool IsUpdate = false;
    using InputType = CostLayerIn;
    using OutputType = LayerIO;

private:
    using ElementType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::Element;
    using DeviceType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::Device;

    using Feedback_ = NSNegativeLogLikelihoodLayer::Feedback_<IsFeedbackOutput>;
public:
    template <typename TIn>
    auto FeedForward(const TIn& p_in)
    {
        const auto& input = p_in.template Get<CostLayerIn>();
        const auto& label = p_in.template Get<CostLayerLabel>();

        using rawType1 = std::decay_t<decltype(input)>;
        using rawType2 = std::decay_t<decltype(label)>;
        static_assert(!std::is_same<rawType1, NullParameter>::value, "Input is invalid");
        static_assert(!std::is_same<rawType2, NullParameter>::value, "Label is invalid");

        Feedback_::Record(label, input, m_label, m_pred);
        return LayerIO::Create().template Set<LayerIO>(NegativeLogLikelihood(label, input));
    }

    template <typename TGrad>
    auto FeedBackward(TGrad&& p_grad)
    {
        return Feedback_::Feedback(std::forward<TGrad>(p_grad), m_label, m_pred);
    }

    void NeutralInvariant()
    {
        if constexpr(IsFeedbackOutput)
        {
            if ((!m_label.empty()) || (!m_pred.empty()))
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
    DataType m_label;
    DataType m_pred;
};
}
