#pragma once

#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>

namespace MetaNN
{
    namespace NSDotLayer
    {
        template <typename TDataCategory>
        constexpr unsigned CategoryWeight = 0;

        template <>
        constexpr unsigned CategoryWeight<CategoryTags::Matrix> = 1;

        template <>
        constexpr unsigned CategoryWeight<CategoryTags::BatchMatrix> = 2;

        template <>
        constexpr unsigned CategoryWeight<CategoryTags::MatrixSequence> = 2;

        template <>
        constexpr unsigned CategoryWeight<CategoryTags::BatchMatrixSequence> = 3;

        template <bool Swap = false, typename TFirstOper, typename TSecondOper>
        auto ShapePrompt(TFirstOper&& first, TSecondOper&& second)
        {
            static_assert(CategoryWeight<DataCategory<TFirstOper>>, "First operand is illegal.");
            static_assert(CategoryWeight<DataCategory<TSecondOper>>, "Second operand is illegal.");

            if constexpr (CategoryWeight<DataCategory<TFirstOper>> == CategoryWeight<DataCategory<TSecondOper>>)
            {
                static_assert(std::is_same_v<DataCategory<TFirstOper>, DataCategory<TSecondOper>>,
                              "Cannot know how to prompt the shape.");
                static_assert(!Swap);
                return std::pair{std::forward<TFirstOper>(first), std::forward<TSecondOper>(second)};
            }
            else if constexpr (CategoryWeight<DataCategory<TFirstOper>> < CategoryWeight<DataCategory<TSecondOper>>)
            {
                return ShapePrompt<true>(std::forward<TFirstOper>(first), std::forward<TSecondOper>(second));
            }
            else if constexpr (std::is_same_v<DataCategory<TFirstOper>, CategoryTags::BatchMatrix> &&
                               std::is_same_v<DataCategory<TSecondOper>, CategoryTags::Matrix>)
            {
                Shape<CategoryTags::BatchMatrix> promptShape{first.Shape().BatchNum(), second.Shape()};
                auto promoted = Duplicate(std::forward<TSecondOper>(second), promptShape);
                if constexpr (Swap)
                {
                    return std::pair{std::move(promoted), std::forward<TFirstOper>(first)};
                }
                else
                {
                    return std::pair{std::forward<TFirstOper>(first), std::move(promoted)};
                }
            }
            else if constexpr (std::is_same_v<DataCategory<TFirstOper>, CategoryTags::MatrixSequence> &&
                               std::is_same_v<DataCategory<TSecondOper>, CategoryTags::Matrix>)
            {
                Shape<CategoryTags::MatrixSequence> promptShape{first.Shape().Length(), second.Shape()};
                auto promoted = Duplicate(std::forward<TSecondOper>(second), promptShape);
                if constexpr (Swap)
                {
                    return std::pair{std::move(promoted), std::forward<TFirstOper>(first)};
                }
                else
                {
                    return std::pair{std::forward<TFirstOper>(first), std::move(promoted)};
                }
            }
            else if constexpr (std::is_same_v<DataCategory<TFirstOper>, CategoryTags::BatchMatrixSequence> &&
                               std::is_same_v<DataCategory<TSecondOper>, CategoryTags::Matrix>)
            {
                Shape<CategoryTags::MatrixSequence> promptShape{first.Shape().SeqLenContainer(), second.Shape()};
                auto promoted = Duplicate(std::forward<TSecondOper>(second), promptShape);
                if constexpr (Swap)
                {
                    return std::pair{std::move(promoted), std::forward<TFirstOper>(first)};
                }
                else
                {
                    return std::pair{std::forward<TFirstOper>(first), std::move(promoted)};
                }
            }
            else
            {
                static_assert(DependencyFalse<TFirstOper, TSecondOper>, "Illegal operand combination.");
            }
        }
    }

    template <typename TInputs, typename TPolicies>
    class DotLayer;
    
    template <>
    struct LayerInputPortSet_<DotLayer<void, void>>
    {
        using type = LayerPortSet<struct LeftOperand, struct RightOperand>;
    };
    
    template <typename TInputs, typename TPolicies>
    class DotLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;
        
        using InputPortSet = LayerInputPortSet<DotLayer>;
        using OutputPortSet = LayerOutputPortSet<DotLayer>;
        using InputMap = TInputs;
        
    private:
        using TLeftOperandFP = typename InputMap::template Find<LeftOperand>;
        using TRightOperandFP = typename InputMap::template Find<RightOperand>;

    public:
        DotLayer(std::string name)
            : m_name(std::move(name))
        {}
        
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            const auto& input1 = LayerTraits::PickItemFromCont<InputMap, LeftOperand>(std::forward<TIn>(p_in));
            const auto& input2 = LayerTraits::PickItemFromCont<InputMap, RightOperand>(std::forward<TIn>(p_in));
            auto [proInput1, proInput2] = NSDotLayer::ShapePrompt(input1, input2);
            auto res = Dot(proInput1, proInput2);
            
            if constexpr (IsFeedbackOutput)
            {
                m_input1.push(input1);
                m_input2.push(input2);
                m_inputShape1.PushDataShape(input1);
                m_inputShape2.PushDataShape(input2);
            }

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

                auto [proInput1, proInput2] = NSDotLayer::ShapePrompt(input1, input2);
                auto grad1 = Dot(grad, Transpose(proInput2));
                auto grad2 = Dot(Transpose(proInput1), grad);

                auto res1 = CollapseOrOmit(std::move(grad1), input1);
                auto res2 = CollapseOrOmit(std::move(grad2), input2);
                m_inputShape1.CheckDataShape(res1);
                m_inputShape2.CheckDataShape(res2);

                LayerTraits::PopoutFromStack(m_input1, m_input2, m_inputShape1, m_inputShape2);
                return LayerInputCont<DotLayer>().template Set<LeftOperand>(std::move(res1))
                                                 .template Set<RightOperand>(std::move(res2));
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