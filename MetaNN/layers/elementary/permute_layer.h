#pragma once

#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/_.h>

namespace MetaNN
{
    template <typename TInputs, typename TPolicies>
    class PermuteLayer
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
        using TDimArray = PickPolicyOjbect<CurLayerPolicy, DimPolicy, DimPolicy::DimArrayValueCate>;

    public:
        PermuteLayer(std::string name)
            : m_name(std::move(name))
        {}

        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto val = LayerTraits::PickItemFromCont<InputMap, LayerInput>(std::forward<TIn>(p_in));
            auto res = Permute<PolicyContainer<TDimArray>>(val);
            
            if constexpr (IsFeedbackOutput)
            {
                m_inputShape.PushDataShape(std::move(val));
            }
            return LayerOutputCont<PermuteLayer>().template Set<LayerOutput>(std::move(res));
        }

        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (!IsFeedbackOutput || RemConstRef<TGrad>::template IsValueEmpty<LayerOutput>)
            {
                if constexpr (IsFeedbackOutput)
                {
                    LayerTraits::PopoutFromStack(m_inputShape);
                }
                return LayerInputCont<PermuteLayer>();
            }
            else
            {
                auto grad = std::forward<TGrad>(p_grad).template Get<LayerOutput>();
                auto res = PermuteInv<PolicyContainer<TDimArray>>(grad);
                m_inputShape.CheckDataShape(res);
                
                LayerTraits::PopoutFromStack(m_inputShape);
                return LayerInputCont<PermuteLayer>().template Set<LayerInput>(std::move(res));
            }
        }

        void NeutralInvariant() const
        {
            if constexpr(IsFeedbackOutput)
            {
                LayerTraits::CheckStackEmpty(m_inputShape);
            }
        }
    private:
        std::string m_name;
        LayerTraits::ShapeChecker<TLayerInputFP,  IsFeedbackOutput> m_inputShape;
    };
    
    template <typename TInputs, typename TPolicies>
    using TransposeLayer = PermuteLayer<TInputs,
                                        ChangePolicy<PDimArrayIs<1, 0>, TPolicies>>;
}