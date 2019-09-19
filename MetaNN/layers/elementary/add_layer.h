#pragma once

#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>
#include <stack>
namespace MetaNN
{
    template <typename TInputs, typename TGrads, typename TPolicies>
    class AddLayer;
    
    template <>
    struct LayerInputPortSet_<AddLayer>
    {
        using type = LayerPortSet<struct LeftOperand, struct RightOperand>;
    };
    
    template <typename TInputs, typename TGrads, typename TPolicies>
    class AddLayer
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
        AddLayer(std::string name)
            : m_name(std::move(name))
        {}

        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            const auto& input1 = LayerTraits::PickItemFromCont<InputMap, LeftOperand>(std::forward<TIn>(p_in));
            const auto& input2 = LayerTraits::PickItemFromCont<InputMap, RightOperand>(std::forward<TIn>(p_in));
            auto proShape = LayerTraits::ShapePromote(input1, input2);
            auto res = DuplicateOrKeep(input1, proShape) + DuplicateOrKeep(input2, proShape);
            
            if constexpr (IsFeedbackOutput)
            {
                m_inputShapeChecker1.PushDataShape(input1);
                m_inputShapeChecker2.PushDataShape(input2);
                m_input1.push(std::move(input1));
                m_input2.push(std::move(input2));
                m_outputShape.PushDataShape(res);
            }

            return LayerOutputCont<AddLayer>().template Set<LayerOutput>(std::move(res));
        }
        
        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (IsFeedbackOutput)
            {
                if ((m_input1.empty()) || (m_input2.empty()))
                {
                    throw std::runtime_error("Cannot feed back in AddLayer");
                }
                
                auto input1 = m_input1.top(); m_input1.pop();
                auto input2 = m_input2.top(); m_input2.pop();
                
                auto grad = LayerTraits::PickItemFromCont<GradMap, LayerOutput>(std::forward<TGrad>(p_grad));
                m_outputShape.CheckDataShapeAndPop(grad);

                auto res1 = CollapseOrOmit(grad, std::move(input1));
                auto res2 = CollapseOrOmit(grad, std::move(input2));
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
                if ((!m_input1.empty()) || (!m_input2.empty()))
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

        LayerTraits::LayerInternalBuf<TLeftOperandFP,  IsFeedbackOutput> m_input1;
        LayerTraits::LayerInternalBuf<TRightOperandFP, IsFeedbackOutput> m_input2;
        
        LayerTraits::ShapeChecker<TLeftOperandFP,  IsFeedbackOutput> m_inputShapeChecker1;
        LayerTraits::ShapeChecker<TRightOperandFP, IsFeedbackOutput> m_inputShapeChecker2;
        LayerTraits::ShapeChecker<TLayerOutputBP,  IsFeedbackOutput> m_outputShape;
    };
}