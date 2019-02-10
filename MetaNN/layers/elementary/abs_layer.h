#pragma once

#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>

namespace MetaNN
{
    struct LayerInput;
    struct LayerOutput;
    
    template <typename TInputs, typename TGrads, typename TPolicies>
    class AbsLayer
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
        AbsLayer(std::string name)
            : m_name(std::move(name))
        {}

        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto val = LayerTraits::PickItemFromCont<InputMap, LayerInput>(std::forward<TIn>(p_in));
            if constexpr (IsFeedbackOutput)
            {
                m_data.push(val);
            }
            return LayerOutputCont<AbsLayer>().template Set<LayerOutput>(Abs(std::move(val)));
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
                auto& input = m_data.top();
                
                auto grad = LayerTraits::PickItemFromCont<GradMap, LayerOutput>(std::forward<TGrad>(p_grad));
                auto res = LayerInputCont<AbsLayer>().template Set<LayerInput>(std::move(grad) * Sign(input));
                m_data.pop();
                return res;
            }
            else
            {
                return LayerInputCont<AbsLayer>();
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
        LayerTraits::LayerInternalBuf<AimInputType, IsFeedbackOutput> m_data;
    };
}