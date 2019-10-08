#pragma once

#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>

namespace MetaNN
{
    template <typename TInputs, typename TPolicies>
    class TransposeLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;
        
        using InputPortSet = LayerInputPortSet<TransposeLayer>;
        using OutputPortSet = LayerOutputPortSet<TransposeLayer>;
        using InputMap = TInputs;
        
    private:
        using TLayerInputFP = typename InputMap::template Find<LayerInput>;

    public:
        TransposeLayer(std::string name)
            : m_name(std::move(name))
        {}

        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto val = LayerTraits::PickItemFromCont<InputMap, LayerInput>(std::forward<TIn>(p_in));
            auto res = Transpose(val);
            
            if constexpr (IsFeedbackOutput)
            {
                m_inputShape.PushDataShape(std::move(val));
            }
            return LayerOutputCont<TransposeLayer>().template Set<LayerOutput>(std::move(res));
        }

        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (!IsFeedbackOutput || RemConstRef<TGrad>::template IsValueEmpty<LayerOutput>)
            {
                if constexpr (IsFeedbackOutput)
                {
                    LayerTraits::PopoutFromStack(m_inputShape);
                }
                return LayerInputCont<TransposeLayer>();
            }
            else
            {
                auto grad = std::forward<TGrad>(p_grad).template Get<LayerOutput>();
                auto res = Transpose(std::move(grad));
                m_inputShape.CheckDataShape(res);
                
                LayerTraits::PopoutFromStack(m_inputShape);
                return LayerInputCont<TransposeLayer>().template Set<LayerInput>(std::move(res));
            }
        }

        void NeutralInvariant() const
        {
            if constexpr(IsFeedbackOutput)
            {
                LayerTraits::CheckStackEmpty(m_inputShape);
            }
        }
    private:
        std::string m_name;
        LayerTraits::ShapeChecker<TLayerInputFP,  IsFeedbackOutput> m_inputShape;
    };
}