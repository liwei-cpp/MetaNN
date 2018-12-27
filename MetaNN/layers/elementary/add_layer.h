#pragma once

#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>
#include <stack>
namespace MetaNN
{
    using AddLayerInput = VarTypeDict<struct AddLayerIn1, struct AddLayerIn2>;
    
    template <typename TInputMap, typename TPolicies>
    class AddLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GeneralPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;
        using InputType = AddLayerInput;
        using OutputType = LayerIO;
        
    private:
        using AimInput1Type = typename TInputMap::template Find<AddLayerIn1>;
        using AimInput2Type = typename TInputMap::template Find<AddLayerIn2>;
        
        using AimInput1ShapeType = RemConstRef<decltype(std::declval<AimInput1Type>().Shape())>;
        using AimInput2ShapeType = RemConstRef<decltype(std::declval<AimInput2Type>().Shape())>;
    public:
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto input1Ori = std::forward<TIn>(p_in).template Get<AddLayerIn1>();
            auto input2Ori = std::forward<TIn>(p_in).template Get<AddLayerIn2>();
            static_assert(!std::is_same_v<decltype(input1Ori), NullParameter>);
            static_assert(!std::is_same_v<decltype(input2Ori), NullParameter>);
            
            auto input1 = LayerTraits::DynamicTransWithFlag<IsDynamic<AimInput1Type>>(std::move(input1Ori));
            auto input2 = LayerTraits::DynamicTransWithFlag<IsDynamic<AimInput2Type>>(std::move(input2Ori));
            
            if constexpr (IsFeedbackOutput)
            {
                m_shape1.push(input1.Shape());
                m_shape2.push(input2.Shape());
            }
            
            auto proShape = LayerTraits::ShapePromote(input1.Shape(), input2.Shape());
            return OutputType::Create().template Set<LayerIO>(Duplicate(input1, proShape) + Duplicate(input2, proShape));
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
                
                auto grad = std::forward<TGrad>(p_grad).template Get<LayerIO>();
                return AddLayerInput::Create().template Set<AddLayerIn1>(Collapse(grad, curShape1))
                                              .template Set<AddLayerIn2>(Collapse(grad, curShape2));
            }
            else
            {
                return AddLayerInput::Create();
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
        LayerTraits::LayerInternalBuf<AimInput1ShapeType, IsFeedbackOutput> m_shape1;
        LayerTraits::LayerInternalBuf<AimInput2ShapeType, IsFeedbackOutput> m_shape2;
    };
}