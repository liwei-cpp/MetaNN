#pragma once

namespace MetaNN
{
    template <typename TInputs, typename TGrads, typename TPolicies>
    class NLLLossLayer;
    
    template <>
    struct LayerInputPortSet_<NLLLossLayer>
    {
        using type = LayerPortSet<struct LayerInput, struct LossLayerWeight>;
    };
    
    template <>
    struct LayerOutputPortSet_<NLLLossLayer>
    {
        using type = LayerPortSet<struct LayerOutput>;
    };
    
    template <typename TInputs, typename TGrads, typename TPolicies>
    class NLLLossLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;

        using InputMap = TInputs;
        using GradMap = TGrads;
        
    private:
        using TLayerInputFP      = typename InputMap::template Find<LayerInput>;
        using TLossLayerWeightFP = typename InputMap::template Find<LossLayerWeight>;
        using TLayerOutputBP     = typename GradMap::template Find<LayerOutput>;

        auto FeedForwardCal(const TLayerInputFP& val, const TLossLayerWeightFP& weight)
        {
            return NLLLoss(weight, val);
        }
    public:
        NLLLossLayer(std::string name)
            : m_name(std::move(name))
        {}

        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto val = LayerTraits::PickItemFromCont<InputMap, LayerInput>(std::forward<TIn>(p_in));
            auto weight = LayerTraits::PickItemFromCont<InputMap, LossLayerWeight>(std::forward<TIn>(p_in));
            auto res = FeedForwardCal(val, weight);
            
            if constexpr (IsFeedbackOutput)
            {
                m_inputShape.PushDataShape(val);
                m_outputShape.PushDataShape(res);
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
                auto input = m_input.top(); m_input.pop();
                auto weight = m_weight.top(); m_weight.pop();

                auto grad = LayerTraits::PickItemFromCont<GradMap, LayerOutput>(std::forward<TGrad>(p_grad));
                m_outputShape.CheckDataShapeAndPop(grad);
                auto res = NLLLossGrad(std::move(grad), std::move(weight), std::move(input));
                m_inputShape.CheckDataShapeAndPop(res);

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
                if ((!m_input.empty()) || (!m_weight.empty()))
                {
                    throw std::runtime_error("NeutralInvariant Fail!");
                }
                m_inputShape.AssertEmpty();
                m_outputShape.AssertEmpty();
            }
        }
    private:
        std::string m_name;
        LayerTraits::LayerInternalBuf<TLayerInputFP,      IsFeedbackOutput> m_input;
        LayerTraits::LayerInternalBuf<TLossLayerWeightFP, IsFeedbackOutput> m_weight;

        LayerTraits::ShapeChecker<TLayerInputFP,  IsFeedbackOutput> m_inputShape;
        LayerTraits::ShapeChecker<TLayerOutputBP, IsFeedbackOutput> m_outputShape;
    };
}