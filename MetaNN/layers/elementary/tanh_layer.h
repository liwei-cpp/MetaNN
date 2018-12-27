#pragma once
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>
#include <stack>

namespace MetaNN
{
    template <typename TInputMap, typename TPolicies>
    class TanhLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GeneralPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;
        using InputType = LayerIO;
        using OutputType = LayerIO;
        using InputTypeMap = TInputMap;
        
    private:
        using AimInputType = typename TInputMap::template Find<LayerIO>;
        
    public:
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto valOri = std::forward<TIn>(p_in).template Get<LayerIO>();
            static_assert(!std::is_same_v<decltype(valOri), NullParameter>);
            auto val = LayerTraits::DynamicTransWithFlag<IsDynamic<AimInputType>>(std::move(valOri));
            static_assert(std::is_same_v<decltype(val), AimInputType>);
            
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
                auto grad = p_grad.template Get<LayerIO>();
                auto& tanhRes = m_data.top();
                auto res = LayerIO::Create().template Set<LayerIO>(TanhGrad(grad, tanhRes));
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
        using TempDataType = decltype(Tanh(std::declval<AimInputType>()));
        LayerTraits::LayerInternalBuf<TempDataType, IsFeedbackOutput> m_data;
    };
}