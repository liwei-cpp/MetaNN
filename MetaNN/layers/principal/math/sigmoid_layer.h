#pragma once

#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/_.h>
#include <stack>

namespace MetaNN
{
    namespace NSSigmoidLayer
    {
        template <bool IsFeedbackOutput, typename TInputType>
        struct InternalDataTypeCalculator
        {
            using type = NullParameter;
        };
        
        template <typename TInputType>
        struct InternalDataTypeCalculator<true, TInputType>
        {
            using OutputType = decltype(Sigmoid(std::declval<TInputType>()));
            using type = LayerTraits::LayerInternalBuf<OutputType, true>;
        };
    }
    
    template <typename TInputs, typename TPolicies>
    class SigmoidLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;

        using InputPortSet = LayerPortSet<struct LayerInput>;
        using OutputPortSet = LayerPortSet<struct LayerOutput>;
        using InputMap = typename std::conditional_t<std::is_same_v<TInputs, NullParameter>,
                                                     EmptyLayerInMap_<InputPortSet>,
                                                     Identity_<TInputs>>::type;
        static_assert(CheckInputMapAvailable_<InputMap, InputPortSet>::value);
        
    private:
        using TLayerInputFP = typename InputMap::template Find<LayerInput>;

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
                m_inputShape.PushDataShape(val);
                m_data.push(res);
            }
            return LayerOutputCont<SigmoidLayer>().template Set<LayerOutput>(std::move(res));
        }

        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (!IsFeedbackOutput || RemConstRef<TGrad>::template IsValueEmpty<LayerOutput>)
            {
                if constexpr (IsFeedbackOutput)
                {
                    LayerTraits::PopoutFromStack(m_data, m_inputShape);
                }
                return LayerInputCont<SigmoidLayer>();
            }
            else
            {
                if (m_data.empty())
                {
                    throw std::runtime_error("Cannot feed back in SigmoidLayer");
                }
                auto input = m_data.top();
                auto grad = std::forward<TGrad>(p_grad).template Get<LayerOutput>();

                auto res = SigmoidGrad(std::move(grad), std::move(input));
                m_inputShape.CheckDataShape(res);
                
                LayerTraits::PopoutFromStack(m_data, m_inputShape);
                return LayerInputCont<SigmoidLayer>().template Set<LayerInput>(std::move(res));
            }
        }

        void NeutralInvariant() const
        {
            if constexpr(IsFeedbackOutput)
            {
                LayerTraits::CheckStackEmpty(m_data, m_inputShape);
            }
        }

    private:
        std::string m_name;
        using InternalDataType = typename NSSigmoidLayer::InternalDataTypeCalculator<IsFeedbackOutput, TLayerInputFP>::type;
        InternalDataType m_data;

        LayerTraits::ShapeChecker<TLayerInputFP,  IsFeedbackOutput> m_inputShape;
    };
}