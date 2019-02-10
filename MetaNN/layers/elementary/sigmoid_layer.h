#pragma once

#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>
#include <stack>

namespace MetaNN
{
    struct LayerInput;
    struct LayerOutput;
    
    template <typename TInputs, typename TGrads, typename TPolicies>
    class SigmoidLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;

        using InputMap = TInputs;
        using GradMap = FillGradMap<TGrads, LayerOutput>;
        
    private:
        using AimInputType = typename InputMap::template Find<LayerInput>;
        
    public:
        SigmoidLayer(std::string name)
            : m_name(std::move(name))
        {}
        
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto val = LayerTraits::PickItemFromCont<InputMap, LayerInput>(std::forward<TIn>(p_in));
            
            auto res = Sigmoid(val);
            if constexpr (IsFeedbackOutput)
            {
                m_data.push(res);
            }
            return LayerOutputCont<SigmoidLayer>().template Set<LayerOutput>(std::move(res));
        }

        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (IsFeedbackOutput)
            {
                if (m_data.empty())
                {
                    throw std::runtime_error("Cannot feed back in SigmoidLayer");
                }
                auto grad = LayerTraits::PickItemFromCont<GradMap, LayerOutput>(std::forward<TGrad>(p_grad));
                
                auto input = m_data.top();
                m_data.pop();
                return LayerInputCont<SigmoidLayer>().template Set<LayerInput>(SigmoidGrad(std::move(grad), std::move(input)));
            }
            else
            {
                return LayerInputCont<SigmoidLayer>();
            }
        }

        void NeutralInvariant() const
        {
            if constexpr(IsFeedbackOutput)
            {
                if (!m_data.empty())
                {
                    throw std::runtime_error("NeutralInvariant Fail!");
                }
            }
        }

    private:
        std::string m_name;
        using TempDataType = decltype(Sigmoid(std::declval<AimInputType>()));
        LayerTraits::LayerInternalBuf<TempDataType, IsFeedbackOutput> m_data;
    };
}