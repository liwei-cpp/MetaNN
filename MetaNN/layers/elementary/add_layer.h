#pragma once

#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>
#include <stack>
namespace MetaNN
{
    template <typename TInputItems, typename TInputGrads, typename TPolicies>
    class AddLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;
        
        using InputContType = BinaryInput;
        using OutputContType = LayerIO;
        
        using InputItemTypes = TInputItems;
        using InputGradTypes = TInputGrads;
        
    private:
        using AimInput1Type = typename InputItemTypes::template Find<LeftOperand>;
        using AimInput2Type = typename InputItemTypes::template Find<RightOperand>;
        
        using AimInput1ShapeType = RemConstRef<decltype(std::declval<AimInput1Type>().Shape())>;
        using AimInput2ShapeType = RemConstRef<decltype(std::declval<AimInput2Type>().Shape())>;

    public:
        AddLayer(std::string name)
            : m_name(std::move(name))
        {}

        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto input1 = LayerTraits::PickItemFromCont<InputItemTypes, LeftOperand>(std::forward<TIn>(p_in));
            auto input2 = LayerTraits::PickItemFromCont<InputItemTypes, RightOperand>(std::forward<TIn>(p_in));
            
            if constexpr (IsFeedbackOutput)
            {
                m_shape1.push(input1.Shape());
                m_shape2.push(input2.Shape());
            }
            
            auto proShape = LayerTraits::ShapePromote(input1.Shape(), input2.Shape());
            return OutputContType::Create().template Set<LayerIO>(Duplicate(input1, proShape) + Duplicate(input2, proShape));
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
                
                auto grad = LayerTraits::PickItemFromCont<InputGradTypes, LayerIO>(std::forward<TGrad>(p_grad));
                
                return BinaryInput::Create().template Set<LeftOperand>(Collapse(grad, curShape1))
                                            .template Set<RightOperand>(Collapse(grad, curShape2));
            }
            else
            {
                return BinaryInput::Create();
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