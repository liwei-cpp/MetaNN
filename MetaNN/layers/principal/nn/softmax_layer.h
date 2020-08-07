#pragma once
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/_.h>
#include <stack>

namespace MetaNN
{
    namespace NSSoftmaxLayer
    {
        template <bool IsSaveData, typename TPolicy, typename TInput>
        struct OutputDataStack_
        {
            using type = NullParameter;
        };
        
        template <typename TPolicy, typename TInput>
        struct OutputDataStack_<true, TPolicy, TInput>
        {
            using ItemType = decltype(Softmax<TPolicy>(std::declval<TInput>()));
            using type = std::stack<ItemType>;
        };
    }
    
    template <typename TInputs, typename TPolicies>
    class SoftmaxLayer
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

        auto FeedForwardCal(const TLayerInputFP& val)
        {
            return Softmax<CurLayerPolicy>(val);
        }
    public:
        SoftmaxLayer(std::string name)
            : m_name(std::move(name))
        {}
        
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto val = LayerTraits::PickItemFromCont<InputMap, LayerInput>(std::forward<TIn>(p_in));
            auto res = Softmax<CurLayerPolicy>(val);
            if constexpr (IsFeedbackOutput)
            {
                m_inputShape.PushDataShape(val);
                m_data.push(res);
            }
            return LayerOutputCont<SoftmaxLayer>().template Set<LayerOutput>(std::move(res));
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
                return LayerInputCont<SoftmaxLayer>();
            }
            else
            {
                if (m_data.empty())
                {
                    throw std::runtime_error("Cannot feed back in SoftmaxLayer");
                }
                auto softmaxRes = m_data.top();
                auto grad = std::forward<TGrad>(p_grad).template Get<LayerOutput>();

                auto res = SoftmaxGrad<CurLayerPolicy>(std::move(grad), std::move(softmaxRes));
                m_inputShape.CheckDataShape(res);
                
                LayerTraits::PopoutFromStack(m_data, m_inputShape);
                return LayerInputCont<SoftmaxLayer>().template Set<LayerInput>(std::move(res));
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
        using TInternalBuf = typename NSSoftmaxLayer::OutputDataStack_<IsFeedbackOutput, CurLayerPolicy, TLayerInputFP>::type;
        TInternalBuf m_data;
        LayerTraits::ShapeChecker<TLayerInputFP,  IsFeedbackOutput> m_inputShape;
    };
}
