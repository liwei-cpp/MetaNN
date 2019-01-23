#pragma once

#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>

namespace MetaNN
{
    template <typename TInputItems, typename TInputGrads, typename TPolicies>
    class AbsLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;
        
        using InputContType = LayerIO;
        using OutputContType = LayerIO;
        
        using InputItemTypes = TInputItems;
        using InputGradTypes = TInputGrads;
        
    private:
        using AimInputType = typename InputItemTypes::template Find<LayerIO>;
        
    public:
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto valOri = std::forward<TIn>(p_in).template Get<LayerIO>();
            static_assert(!std::is_same_v<decltype(valOri), NullParameter>);

            auto val = LayerTraits::DynamicTransWithFlag<IsDynamic<AimInputType>>(std::move(valOri));
            static_assert(std::is_same_v<decltype(val), AimInputType>);

            if constexpr (IsFeedbackOutput)
            {
                m_data.push(val);
            }
            return LayerIO::Create().template Set<LayerIO>(Abs(std::move(val)));
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
                
                auto gradOri = std::forward<TGrad>(p_grad).template Get<LayerIO>();
                static_assert(!std::is_same_v<decltype(gradOri), NullParameter>);
                using AimGradType = typename InputGradTypes::template Find<LayerIO>;
                static_assert(!std::is_same_v<AimGradType, NullParameter>);
                auto grad = LayerTraits::DynamicTransWithFlag<IsDynamic<AimGradType>>(std::move(gradOri));
                
                auto& input = m_data.top();
                auto res = LayerIO::Create().template Set<LayerIO>(std::move(grad) * Sign(input));
                m_data.pop();
                return res;
            }
            else
            {
                return LayerIO::Create();
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
        LayerTraits::LayerInternalBuf<AimInputType, IsFeedbackOutput> m_data;
    };
}