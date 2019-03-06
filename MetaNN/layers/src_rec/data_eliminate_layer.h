#pragma once

#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>

namespace MetaNN
{
    namespace NSDataEliminateLayer
    {
        template <typename TCategory, typename TElement, typename TDevice>
        auto CreateGradRes(const Shape<TCategory>& shape)
        {
            return ZeroData<TCategory, TElement, TDevice>{shape};
        }
    }
    
    struct LayerInput;
    
    template <typename TInputs, typename TGrads, typename TPolicies>
    class DataEliminateLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = false;

    public:
        using InputMap = TInputs;
        using GradMap = LayerIOMap<>;

    private:
        using TLayerInputFP = typename InputMap::template Find<LayerInput>;

    public:
        DataEliminateLayer(std::string name)
            : m_name(std::move(name)) {}

        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto val = LayerTraits::PickItemFromCont<InputMap, LayerInput>(std::forward<TIn>(p_in));
            if constexpr (IsFeedbackOutput)
            {
                m_inputShapeStack.push(val.Shape());
            }
            return LayerOutputCont<DataEliminateLayer>();
        }

        template <typename TGrad>
        auto FeedBackward(TGrad&&)
        {
            if constexpr (IsFeedbackOutput)
            {
                if (m_inputShapeStack.empty())
                {
                    throw std::runtime_error("Cannot backward from Data eliminate layer");
                }
                auto shape = m_inputShapeStack.top();
                m_inputShapeStack.pop();
                
                auto res
                    = NSDataEliminateLayer::CreateGradRes<typename TLayerInputFP::CategoryTag,
                                                          typename TLayerInputFP::ElementType,
                                                          typename TLayerInputFP::DeviceType>(shape);
                return LayerInputCont<DataEliminateLayer>().template Set<LayerInput>(std::move(res));
            }
            else
            {
                return LayerInputCont<DataEliminateLayer>();
            }
        }
        
        void NeutralInvariant() const
        {
            if constexpr(IsFeedbackOutput)
            {
                if (!m_inputShapeStack.empty())
                {
                    throw std::runtime_error("NeutralInvariant Fail!");
                }
            }
        }

    private:
        std::string m_name;
        LayerTraits::LayerInternalBuf<ShapeType<TLayerInputFP>, IsFeedbackOutput> m_inputShapeStack;
    };
}