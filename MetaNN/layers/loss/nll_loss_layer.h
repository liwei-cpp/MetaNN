#pragma once

namespace MetaNN
{
    template <typename TInputs, typename TPolicies>
    class NLLLossLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;

        using InputPortSet = LayerPortSet<struct LayerInput, struct LossLayerWeight>;
        using OutputPortSet = LayerPortSet<struct LayerOutput>;
        using InputMap = typename std::conditional_t<std::is_same_v<TInputs, NullParameter>,
                                                     EmptyLayerInMap_<InputPortSet>,
                                                     Identity_<TInputs>>::type;
        static_assert(CheckInputMapAvailable_<InputMap, InputPortSet>::value);
        
    private:
        using TLayerInputFP      = typename InputMap::template Find<LayerInput>;
        using TLossLayerWeightFP = typename InputMap::template Find<LossLayerWeight>;

    public:
        NLLLossLayer(std::string name)
            : m_name(std::move(name))
        {}

        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto val = LayerTraits::PickItemFromCont<InputMap, LayerInput>(std::forward<TIn>(p_in));
            auto weight = LayerTraits::PickItemFromCont<InputMap, LossLayerWeight>(std::forward<TIn>(p_in));
            auto res = NLLLoss(weight, val);
            
            if constexpr (IsFeedbackOutput)
            {
                m_inputShape.PushDataShape(val);
                m_input.push(std::move(val));
                m_weight.push(std::move(weight));
            }
            return LayerOutputCont<NLLLossLayer>().template Set<LayerOutput>(std::move(res));
        }
        
        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (IsFeedbackOutput)
            {
                if ((m_input.empty()) || (m_weight.empty()))
                {
                    throw std::runtime_error("Cannot feed back in NLL loss layer");
                }
                auto input = m_input.top(); auto weight = m_weight.top();
                auto grad = std::forward<TGrad>(p_grad).template Get<LayerOutput>();
                auto res = NLLLossGrad(std::move(grad), std::move(weight), std::move(input));
                m_inputShape.CheckDataShape(res);
                
                LayerTraits::PopoutFromStack(m_input, m_weight, m_inputShape);
                return LayerInputCont<NLLLossLayer>().template Set<LayerInput>(std::move(res));
            }
            else
            {
                return LayerInputCont<NLLLossLayer>();
            }
        }

        void NeutralInvariant() const
        {
            if constexpr(IsFeedbackOutput)
            {
                LayerTraits::CheckStackEmpty(m_input, m_weight, m_inputShape);
            }
        }
    private:
        std::string m_name;
        LayerTraits::LayerInternalBuf<TLayerInputFP,      IsFeedbackOutput> m_input;
        LayerTraits::LayerInternalBuf<TLossLayerWeightFP, IsFeedbackOutput> m_weight;

        LayerTraits::ShapeChecker<TLayerInputFP,  IsFeedbackOutput> m_inputShape;
    };
}