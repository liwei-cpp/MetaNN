#pragma once

#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/_.h>

namespace MetaNN
{
    namespace NSDotLayer
    {
        template <typename TIndexArr, size_t uModDim>
        struct ModeDim2PermuteArrHelper_;

        template <size_t... I, size_t uModDim>
        struct ModeDim2PermuteArrHelper_<std::index_sequence<I...>, uModDim>
        {
            constexpr static size_t dimLen = sizeof...(I);
            using type = PDimArrayIs<((I + uModDim) % dimLen)...>;
        };

        template <size_t uDim, size_t uModDim>
        struct ModeDim2PermuteArr_
        {
            using type = typename ModeDim2PermuteArrHelper_<std::make_index_sequence<uDim>, uModDim>::type;
        };
    }
    
    template <typename TInputs, typename TPolicies>
    class DotLayer
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
        
        constexpr static size_t modDimNum = PolicySelect<DimPolicy, CurLayerPolicy>::ModifyDimNum;

    public:
        DotLayer(std::string name)
            : m_name(std::move(name))
        {}
        
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            const auto& input1 = LayerTraits::PickItemFromCont<InputMap, LeftOperand>(std::forward<TIn>(p_in));
            const auto& input2 = LayerTraits::PickItemFromCont<InputMap, RightOperand>(std::forward<TIn>(p_in));
            auto res = Dot<CurLayerPolicy>(input1, input2);
            
            if constexpr (IsFeedbackOutput)
            {
                m_input1.push(input1);
                m_input2.push(input2);
                m_inputShape1.PushDataShape(input1);
                m_inputShape2.PushDataShape(input2);
            }
            
            static_assert(DataCategory<decltype(res)>::DimNum ==
                          DataCategory<decltype(input1)>::DimNum + DataCategory<decltype(input2)>::DimNum - modDimNum * 2);

            return LayerOutputCont<DotLayer>().template Set<LayerOutput>(std::move(res));
        }
        
        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (!IsFeedbackOutput || RemConstRef<TGrad>::template IsValueEmpty<LayerOutput>)
            {
                if constexpr (IsFeedbackOutput)
                {
                    LayerTraits::PopoutFromStack(m_input1, m_input2, m_inputShape1, m_inputShape2);
                }
                return LayerInputCont<DotLayer>();
            }
            else
            {
                if ((m_input1.empty()) || (m_input2.empty()))
                {
                    throw std::runtime_error("Cannot feed back in DotLayer");
                }

                auto input1 = m_input1.top(); auto input2 = m_input2.top();
                auto grad = std::forward<TGrad>(p_grad).template Get<LayerOutput>();
                static_assert(DataCategory<decltype(grad)>::DimNum ==
                              DataCategory<TLeftOperandFP>::DimNum + DataCategory<TRightOperandFP>::DimNum - modDimNum * 2,
                              "Grad has mismatch dimension.");
                
                using PPermute1 = typename NSDotLayer::ModeDim2PermuteArr_<DataCategory<TRightOperandFP>::DimNum, modDimNum>::type;
                constexpr size_t grad1DotDim = DataCategory<TRightOperandFP>::DimNum - modDimNum;
                auto grad1 = Dot<PolicyContainer<PModifyDimNumIs<grad1DotDim>>>(grad, Permute<PolicyContainer<PPermute1>>(input2));

                constexpr size_t grad2DotDim = DataCategory<TLeftOperandFP>::DimNum - modDimNum;
                using PPermute2 = typename NSDotLayer::ModeDim2PermuteArr_<DataCategory<TLeftOperandFP>::DimNum, grad2DotDim>::type;
                auto grad2 = Dot<PolicyContainer<PModifyDimNumIs<grad2DotDim>>>(Permute<PolicyContainer<PPermute2>>(input1), grad);

                m_inputShape1.CheckDataShape(grad1);
                m_inputShape2.CheckDataShape(grad2);

                LayerTraits::PopoutFromStack(m_input1, m_input2, m_inputShape1, m_inputShape2);
                return LayerInputCont<DotLayer>().template Set<LeftOperand>(std::move(grad1))
                                                 .template Set<RightOperand>(std::move(grad2));
            }
        }
        
        void NeutralInvariant()
        {
            if constexpr(IsFeedbackOutput)
            {
                LayerTraits::CheckStackEmpty(m_input1, m_input2, m_inputShape1, m_inputShape2);
            }
        }
    private:
        std::string m_name;
        LayerTraits::LayerInternalBuf<TLeftOperandFP, IsFeedbackOutput> m_input1;
        LayerTraits::LayerInternalBuf<TRightOperandFP, IsFeedbackOutput> m_input2;

        LayerTraits::ShapeChecker<TLeftOperandFP,  IsFeedbackOutput> m_inputShape1;
        LayerTraits::ShapeChecker<TRightOperandFP, IsFeedbackOutput> m_inputShape2;
    };
}