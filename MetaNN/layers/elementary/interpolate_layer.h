#pragma once
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>

namespace MetaNN
{
    struct InterpolateLayerWeight1; struct InterpolateLayerWeight2; struct InterpolateLayerLambda;
    struct LayerOutput;

    template <typename TInputs, typename TGrads, typename TPolicies>
    class InterpolateLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;
        
        using InputMap = TInputs;
        using GradMap = FillGradMap<TGrads, LayerOutput>;
        
    private:
        using Input1Type = typename InputMap::template Find<InterpolateLayerWeight1>;
        using Input2Type = typename InputMap::template Find<InterpolateLayerWeight2>;
        using InputLambdaType = typename InputMap::template Find<InterpolateLayerLambda>;

    public:
        InterpolateLayer(std::string name)
            : m_name(std::move(name))
        {}
        
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto input1 = LayerTraits::PickItemFromCont<InputMap, InterpolateLayerWeight1>(std::forward<TIn>(p_in));
            auto input2 = LayerTraits::PickItemFromCont<InputMap, InterpolateLayerWeight2>(std::forward<TIn>(p_in));
            auto lambda = LayerTraits::PickItemFromCont<InputMap, InterpolateLayerLambda>(std::forward<TIn>(p_in));

            if constexpr (IsFeedbackOutput)
            {
                m_input1Stack.push(input1);
                m_input2Stack.push(input2);
                m_lambdaStack.push(lambda);
            }
            
            auto proShape = LayerTraits::ShapePromote(input1.Shape(), input2.Shape(), lambda.Shape());
            auto res = Interpolate(Duplicate(std::move(input1), proShape),
                                   Duplicate(std::move(input2), proShape),
                                   Duplicate(std::move(lambda), proShape));
            return LayerOutputCont<InterpolateLayer>().template Set<LayerOutput>(std::move(res));
        }

        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (IsFeedbackOutput)
            {
                if ((m_input1Stack.empty()) || (m_input2Stack.empty()) || (m_lambdaStack.empty()))
                {
                    throw std::runtime_error("Cannot do FeedBackward for InterpolateLayer");
                }
                auto grad = LayerTraits::PickItemFromCont<GradMap, LayerOutput>(std::forward<TGrad>(p_grad));
                auto curLambda = m_lambdaStack.top();
                auto curInput1 = m_input1Stack.top();
                auto curInput2 = m_input2Stack.top();
                m_lambdaStack.pop();
                m_input1Stack.pop();
                m_input2Stack.pop();

                auto res2 = grad * Duplicate(1 - curLambda, grad.Shape());
                auto res1 = grad * Duplicate(curLambda, grad.Shape());
                auto resLambda = grad * (Duplicate(curInput1, grad.Shape()) - Duplicate(curInput2, grad.Shape()));
                return LayerInputCont<InterpolateLayer>()
                    .template Set<InterpolateLayerWeight1>(Collapse(std::move(res1), curInput1.Shape()))
                    .template Set<InterpolateLayerWeight2>(Collapse(std::move(res2), curInput2.Shape()))
                    .template Set<InterpolateLayerLambda>(Collapse(std::move(resLambda), curLambda.Shape()));
            }
            else
            {
                return LayerInputCont<InterpolateLayer>();
            }
        }

        void NeutralInvariant()
        {
            if constexpr(IsFeedbackOutput)
            {
                if ((!m_input1Stack.empty()) || (!m_input2Stack.empty()) || (!m_lambdaStack.empty()))
                {
                    throw std::runtime_error("NeutralInvariant Fail!");
                }
            }
        }
    private:
        std::string m_name;
        LayerTraits::LayerInternalBuf<Input1Type, IsFeedbackOutput> m_input1Stack;
        LayerTraits::LayerInternalBuf<Input2Type, IsFeedbackOutput> m_input2Stack;
        LayerTraits::LayerInternalBuf<InputLambdaType, IsFeedbackOutput> m_lambdaStack;
    };
}