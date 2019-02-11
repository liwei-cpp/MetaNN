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
        using TLayerInputFP = typename InputMap::template Find<LayerInput>;
        using TLayerOutputBP = typename GradMap::template Find<LayerOutput>;
        
        template <typename TVal>
        auto FeedForwardCal(const TVal& val)
        {
            return Abs(val);
        }

    public:
        AbsLayer(std::string name)
            : m_name(std::move(name))
        {}

        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto val = LayerTraits::PickItemFromCont<InputMap, LayerInput>(std::forward<TIn>(p_in));
            auto res = FeedForwardCal(val);
            
            if constexpr (IsFeedbackOutput)
            {
                m_inputShape.push(val.Shape());
                m_outputShape.push(res.Shape());
                m_data.push(std::move(val));
            }
            return LayerOutputCont<AbsLayer>().template Set<LayerOutput>(std::move(res));
        }

        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (IsFeedbackOutput)
            {
                if ((m_data.empty()) || (m_outputShape.empty()))
                {
                    throw std::runtime_error("Cannot feed back in AbsLayer");
                }
                auto input = m_data.top();
                m_data.pop();

                auto grad = LayerTraits::PickItemFromCont<GradMap, LayerOutput>(std::forward<TGrad>(p_grad));
                LayerTraits::ShapeCheck(grad, m_outputShape);
                auto res = std::move(grad) * Sign(std::move(input));
                LayerTraits::ShapeCheck(res, m_inputShape);
                m_outputShape.pop(); m_inputShape.pop();

                return LayerInputCont<AbsLayer>().template Set<LayerInput>(std::move(res));
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
                if ((!m_data.empty()) || (!m_outputShape.empty()) || (!m_inputShape.empty()))
                {
                    throw std::runtime_error("NeutralInvariant Fail!");
                }
            }
        }
    private:
        std::string m_name;
        LayerTraits::LayerInternalBuf<TLayerInputFP, IsFeedbackOutput> m_data;
        LayerTraits::LayerInternalBuf<ShapeType<TLayerInputFP>, IsFeedbackOutput> m_inputShape;
        LayerTraits::LayerInternalBuf<ShapeType<TLayerOutputBP>, IsFeedbackOutput> m_outputShape;
    };
}