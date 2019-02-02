#pragma once
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>

namespace MetaNN
{
    using ElementMulLayerInput = VarTypeDict<struct ElementMulLayerIn1, struct ElementMulLayerIn2>;
    
    template <typename TInputItems, typename TInputGrads, typename TPolicies>
    class ElementMulLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;
        
        using InputContType = ElementMulLayerInput;
        using OutputContType = LayerIO;
        
        using InputItemTypes = TInputItems;
        using InputGradTypes = TInputGrads;
        
    private:
        using AimInput1Type = typename InputItemTypes::template Find<ElementMulLayerIn1>;
        using AimInput2Type = typename InputItemTypes::template Find<ElementMulLayerIn2>;
        
    public:
        ElementMulLayer(std::string name)
            : m_name(std::move(name))
        {}
        
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto input1 = LayerTraits::PickItemFromCont<InputItemTypes, ElementMulLayerIn1>(std::forward<TIn>(p_in));
            auto input2 = LayerTraits::PickItemFromCont<InputItemTypes, ElementMulLayerIn2>(std::forward<TIn>(p_in));
            
            if constexpr (IsFeedbackOutput)
            {
                m_input1.push(input1);
                m_input2.push(input2);
            }
            
            auto proShape = LayerTraits::ShapePromote(input1.Shape(), input2.Shape());
            return OutputContType::Create().template Set<LayerIO>(Duplicate(std::move(input1), proShape) *
                                                                  Duplicate(std::move(input2), proShape));
        }
        
        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (IsFeedbackOutput)
            {
                if ((m_input1.empty()) || (m_input2.empty()))
                {
                    throw std::runtime_error("Cannot feed back in ElementMulLayer");
                }
                
                auto input1 = m_input1.top();
                auto input2 = m_input2.top();
                m_input1.pop();
                m_input2.pop();
                
                auto grad = LayerTraits::PickItemFromCont<InputGradTypes, LayerIO>(std::forward<TGrad>(p_grad));
                
                auto shape1 = input1.Shape();
                auto shape2 = input2.Shape();
                
                auto grad1 = grad * Duplicate(input1, grad.Shape());
                auto grad2 = grad * Duplicate(input2, grad.Shape());
                return ElementMulLayerInput::Create().template Set<ElementMulLayerIn1>(Collapse(std::move(grad2), shape1))
                                                     .template Set<ElementMulLayerIn2>(Collapse(std::move(grad1), shape2));
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
                if ((!m_input1.empty()) || (!m_input2.empty()))
                {
                    throw std::runtime_error("NeutralInvariant Fail!");
                }
            }
        }
    private:
        std::string m_name;
        LayerTraits::LayerInternalBuf<AimInput1Type, IsFeedbackOutput> m_input1;
        LayerTraits::LayerInternalBuf<AimInput2Type, IsFeedbackOutput> m_input2;
    };
}
