#pragma once
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>

namespace MetaNN
{
    template <typename TInputs, typename TGrads, typename TPolicies>
    class MultiplyLayer;
    
    template <>
    struct LayerInputPortSet_<MultiplyLayer>
    {
        using type = LayerPortSet<struct LeftOperand, struct RightOperand>;
    };
    
    template <typename TInputs, typename TGrads, typename TPolicies>
    class MultiplyLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;
        
        using InputMap = TInputs;
        using GradMap = TGrads;
        
    private:
        using TLeftOperandFP = typename InputMap::template Find<LeftOperand>;
        using TRightOperandFP = typename InputMap::template Find<RightOperand>;
        using TLayerOutputBP = typename GradMap::template Find<LayerOutput>;

    public:
        MultiplyLayer(std::string name)
            : m_name(std::move(name))
        {}
        
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            const auto& input1 = LayerTraits::PickItemFromCont<InputMap, LeftOperand>(std::forward<TIn>(p_in));
            const auto& input2 = LayerTraits::PickItemFromCont<InputMap, RightOperand>(std::forward<TIn>(p_in));
            auto proShape = LayerTraits::ShapePromote(input1, input2);
            auto res = DuplicateOrKeep(input1, proShape) * DuplicateOrKeep(input2, proShape);
            
            if constexpr (IsFeedbackOutput)
            {
                m_input1.push(input1);
                m_input2.push(input2);
                m_inputShape1.PushDataShape(input1);
                m_inputShape2.PushDataShape(input2);
                m_outputShape.PushDataShape(res);
            }

            return LayerOutputCont<MultiplyLayer>().template Set<LayerOutput>(std::move(res));
        }
        
        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (IsFeedbackOutput)
            {
                if ((m_input1.empty()) || (m_input2.empty()))
                {
                    throw std::runtime_error("Cannot feed back in MultiplyLayer");
                }
                
                auto input1 = m_input1.top();
                auto input2 = m_input2.top();
                m_input1.pop();
                m_input2.pop();

                auto grad = LayerTraits::PickItemFromCont<GradMap, LayerOutput>(std::forward<TGrad>(p_grad));
                m_outputShape.CheckDataShapeAndPop(grad);

                auto grad1 = grad * DuplicateOrKeep(input1, grad.Shape());
                auto grad2 = grad * DuplicateOrKeep(input2, grad.Shape());
                auto res1 = CollapseOrOmit(std::move(grad2), input1);
                auto res2 = CollapseOrOmit(std::move(grad1), input2);
                m_inputShape1.CheckDataShapeAndPop(res1);
                m_inputShape2.CheckDataShapeAndPop(res2);
                return LayerInputCont<MultiplyLayer>().template Set<LeftOperand>(std::move(res1))
                                                      .template Set<RightOperand>(std::move(res2));
            }
            else
            {
                return LayerInputCont<MultiplyLayer>();
            }
        }
        
        void NeutralInvariant()
        {
            if constexpr(IsFeedbackOutput)
            {
                if ((!m_input1.empty()) || (!m_input2.empty()))
                {
                    throw std::runtime_error("NeutralInvariant Fail!");
                }
                m_inputShape1.AssertEmpty();
                m_inputShape2.AssertEmpty();
                m_outputShape.AssertEmpty();
            }
        }
    private:
        std::string m_name;
        LayerTraits::LayerInternalBuf<TLeftOperandFP, IsFeedbackOutput> m_input1;
        LayerTraits::LayerInternalBuf<TRightOperandFP, IsFeedbackOutput> m_input2;

        LayerTraits::ShapeChecker<TLeftOperandFP,  IsFeedbackOutput> m_inputShape1;
        LayerTraits::ShapeChecker<TRightOperandFP, IsFeedbackOutput> m_inputShape2;
        LayerTraits::ShapeChecker<TLayerOutputBP,  IsFeedbackOutput> m_outputShape;
    };
}
