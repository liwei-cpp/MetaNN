#pragma once

#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>
#include <stack>
namespace MetaNN
{
    struct LeftOperand; struct RightOperand;
    struct LayerOutput;

    template <typename TInputs, typename TGrads, typename TPolicies>
    class AddLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;
        
        using InputMap = TInputs;
        using GradMap = FillGradMap<TGrads, LayerOutput>;
        
    private:
        using TLeftOperandFP = typename InputMap::template Find<LeftOperand>;
        using TRightOperandFP = typename InputMap::template Find<RightOperand>;
        using TLayerOutputBP = typename GradMap::template Find<LayerOutput>;
        
        auto FeedForwardCal(const TLeftOperandFP& val1, const TRightOperandFP& val2)
        {
            auto proShape = LayerTraits::ShapePromote(val1, val2);
            return Duplicate(val1, proShape) + Duplicate(val2, proShape);
        }
    public:
        AddLayer(std::string name)
            : m_name(std::move(name))
        {}

        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            const auto& input1 = LayerTraits::PickItemFromCont<InputMap, LeftOperand>(std::forward<TIn>(p_in));
            const auto& input2 = LayerTraits::PickItemFromCont<InputMap, RightOperand>(std::forward<TIn>(p_in));
            auto res = FeedForwardCal(input1, input2);
            
            if constexpr (IsFeedbackOutput)
            {
                m_shape1.push(input1.Shape());
                m_shape2.push(input2.Shape());
                
                m_inputShapeChecker1.PushDataShape(input1);
                m_inputShapeChecker2.PushDataShape(input2);
                m_outputShape.PushDataShape(res);
            }

            return LayerOutputCont<AddLayer>().template Set<LayerOutput>(std::move(res));
        }
        
        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (IsFeedbackOutput)
            {
                if ((m_shape1.empty()) || (m_shape2.empty()))
                {
                    throw std::runtime_error("Cannot feed back in AddLayer");
                }
                
                auto curShape1 = m_shape1.top(); m_shape1.pop();
                auto curShape2 = m_shape2.top(); m_shape2.pop();
                
                auto grad = LayerTraits::PickItemFromCont<GradMap, LayerOutput>(std::forward<TGrad>(p_grad));
                m_outputShape.CheckDataShapeAndPop(grad);

                auto res1 = Collapse(grad, curShape1);
                auto res2 = Collapse(grad, curShape2);
                m_inputShapeChecker1.CheckDataShapeAndPop(res1);
                m_inputShapeChecker2.CheckDataShapeAndPop(res2);

                return LayerInputCont<AddLayer>().template Set<LeftOperand>(std::move(res1))
                                                 .template Set<RightOperand>(std::move(res2));
            }
            else
            {
                return LayerInputCont<AddLayer>();
            }
        }
        
        void NeutralInvariant() const
        {
            if constexpr(IsFeedbackOutput)
            {
                if ((!m_shape1.empty()) || (!m_shape2.empty()))
                {
                    throw std::runtime_error("NeutralInvariant Fail!");
                }
                m_inputShapeChecker1.AssertEmpty();
                m_inputShapeChecker2.AssertEmpty();
                m_outputShape.AssertEmpty();
            }
        }
    private:
        std::string m_name;

        LayerTraits::LayerInternalBuf<ShapeType<TLeftOperandFP>, IsFeedbackOutput> m_shape1;
        LayerTraits::LayerInternalBuf<ShapeType<TRightOperandFP>, IsFeedbackOutput> m_shape2;
        
        LayerTraits::ShapeChecker<TLeftOperandFP,  IsFeedbackOutput> m_inputShapeChecker1;
        LayerTraits::ShapeChecker<TRightOperandFP, IsFeedbackOutput> m_inputShapeChecker2;
        LayerTraits::ShapeChecker<TLayerOutputBP,  IsFeedbackOutput> m_outputShape;
    };
}