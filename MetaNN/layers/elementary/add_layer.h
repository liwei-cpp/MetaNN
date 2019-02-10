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
        using AimInput1Type = typename InputMap::template Find<LeftOperand>;
        using AimInput2Type = typename InputMap::template Find<RightOperand>;
        
        using AimInput1ShapeType = RemConstRef<decltype(std::declval<AimInput1Type>().Shape())>;
        using AimInput2ShapeType = RemConstRef<decltype(std::declval<AimInput2Type>().Shape())>;

    public:
        AddLayer(std::string name)
            : m_name(std::move(name))
        {}

        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto input1 = LayerTraits::PickItemFromCont<InputMap, LeftOperand>(std::forward<TIn>(p_in));
            auto input2 = LayerTraits::PickItemFromCont<InputMap, RightOperand>(std::forward<TIn>(p_in));
            
            if constexpr (IsFeedbackOutput)
            {
                m_shape1.push(input1.Shape());
                m_shape2.push(input2.Shape());
            }
            
            auto proShape = LayerTraits::ShapePromote(input1.Shape(), input2.Shape());
            return LayerOutputCont<AddLayer>().template Set<LayerOutput>(Duplicate(input1, proShape) + Duplicate(input2, proShape));
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
                
                auto curShape1 = m_shape1.top();
                auto curShape2 = m_shape2.top();
                m_shape1.pop();
                m_shape2.pop();
                
                auto grad = LayerTraits::PickItemFromCont<GradMap, LayerOutput>(std::forward<TGrad>(p_grad));
                
                return LayerInputCont<AddLayer>().template Set<LeftOperand>(Collapse(grad, curShape1))
                                                 .template Set<RightOperand>(Collapse(grad, curShape2));
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
            }
        }
    private:
        std::string m_name;
        LayerTraits::LayerInternalBuf<AimInput1ShapeType, IsFeedbackOutput> m_shape1;
        LayerTraits::LayerInternalBuf<AimInput2ShapeType, IsFeedbackOutput> m_shape2;
    };
}