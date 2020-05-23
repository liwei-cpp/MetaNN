#pragma once

#include <MetaNN/layers/facilities/_.h>
#include <MetaNN/policies/_.h>
#include <stack>
namespace MetaNN
{
    template <typename TInputs, typename TPolicies>
    class SubstractLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;
        
        using InputPortSet = LayerPortSet<struct LeftOperand, struct RightOperand>;
        using OutputPortSet = LayerPortSet<struct LayerOutput>;
        using InputMap = typename std::conditional_t<std::is_same_v<TInputs, NullParameter>,
                                                     EmptyLayerInMap_<InputPortSet>,
                                                     Identity_<TInputs>>::type;
        static_assert(CheckInputMapAvailable_<InputMap, InputPortSet>::value);

    private:
        using TLeftOperandFP = typename InputMap::template Find<LeftOperand>;
        using TRightOperandFP = typename InputMap::template Find<RightOperand>;
        
    public:
        SubstractLayer(std::string name)
            : m_name(std::move(name))
        {}

        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            const auto& input1 = LayerTraits::PickItemFromCont<InputMap, LeftOperand>(std::forward<TIn>(p_in));
            const auto& input2 = LayerTraits::PickItemFromCont<InputMap, RightOperand>(std::forward<TIn>(p_in));
            
            if constexpr (IsFeedbackOutput)
            {
                m_inputShapeChecker1.PushDataShape(input1);
                m_inputShapeChecker2.PushDataShape(input2);
            }

            return LayerOutputCont<SubstractLayer>().template Set<LayerOutput>(input1 - input2);
        }
        
        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (!IsFeedbackOutput || RemConstRef<TGrad>::template IsValueEmpty<LayerOutput>)
            {
                if constexpr (IsFeedbackOutput)
                {
                    LayerTraits::PopoutFromStack(m_inputShapeChecker1, m_inputShapeChecker2);
                }
                return LayerInputCont<SubstractLayer>();
            }
            else
            {
                auto grad = std::forward<TGrad>(p_grad).template Get<LayerOutput>();

                auto res1 = LayerTraits::Collapse<TLeftOperandFP>(grad);
                auto res2 = LayerTraits::Collapse<TRightOperandFP>(-grad);
                m_inputShapeChecker1.CheckDataShape(res1);
                m_inputShapeChecker2.CheckDataShape(res2);
                
                LayerTraits::PopoutFromStack(m_inputShapeChecker1, m_inputShapeChecker2);
                return LayerInputCont<SubstractLayer>().template Set<LeftOperand>(std::move(res1))
                                                       .template Set<RightOperand>(std::move(res2));
            }
        }
        
        void NeutralInvariant() const
        {
            if constexpr(IsFeedbackOutput)
            {
                LayerTraits::CheckStackEmpty(m_inputShapeChecker1, m_inputShapeChecker2);
            }
        }
    private:
        std::string m_name;

        LayerTraits::ShapeChecker<TLeftOperandFP,  IsFeedbackOutput> m_inputShapeChecker1;
        LayerTraits::ShapeChecker<TRightOperandFP, IsFeedbackOutput> m_inputShapeChecker2;
    };
}