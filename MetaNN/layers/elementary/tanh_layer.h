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
    class TanhLayer
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
            auto val = LayerTraits::PickItemFromCont<InputItemTypes, LayerIO>(std::forward<TIn>(p_in));
            
            auto res = Tanh(val);
            if constexpr (IsFeedbackOutput)
            {
                m_data.push(res);
            }
            return LayerIO::Create().template Set<LayerIO>(std::move(res));
        }

        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (IsFeedbackOutput)
            {
                if (m_data.empty())
                {
                    throw std::runtime_error("Cannot feed back in TanhLayer");
                }
                auto grad = LayerTraits::PickItemFromCont<InputGradTypes, LayerIO>(std::forward<TGrad>(p_grad));
                
                auto tanhRes = m_data.top();
                m_data.pop();
                auto res = LayerIO::Create().template Set<LayerIO>(TanhGrad(std::move(grad), std::move(tanhRes)));
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
        using TempDataType = decltype(Tanh(std::declval<AimInputType>()));
        LayerTraits::LayerInternalBuf<TempDataType, IsFeedbackOutput> m_data;
    };
}