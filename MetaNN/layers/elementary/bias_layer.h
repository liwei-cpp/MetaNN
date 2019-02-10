#pragma once

#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>
#include <MetaNN/model/param_initializer/facilities/traits.h>

#include <iostream>
namespace MetaNN
{
    namespace NSBiasLayer
    {
        template <typename TAimGrad, typename TBiasShape, bool IsUpdate>
        struct GradStackType_
        {
            using type = NullParameter;
        };
        
        template <typename TAimGrad, typename TBiasShape>
        struct GradStackType_<TAimGrad, TBiasShape, true>
        {
            using TParamGradStackItem = decltype(Collapse(std::declval<TAimGrad>(), std::declval<TBiasShape>()));
            using type = LayerTraits::LayerInternalBuf<TParamGradStackItem, true>;
        };
    }
    
    struct LayerInput;
    struct LayerOutput;
    
    template <typename TInputs, typename TGrads, typename TPolicies>
    class BiasLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;
    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = PolicySelect<GradPolicy, CurLayerPolicy>::IsUpdate;
        
        using InputMap = TInputs;
        using GradMap = FillGradMap<TGrads, LayerOutput>;

    private:
        using ParamCategory = typename PolicySelect<ParamPolicy, CurLayerPolicy>::ParamType;
        using AimInputType = typename InputMap::template Find<LayerInput>;
        using ParamType = PrincipalDataType<ParamCategory,
                                            typename AimInputType::ElementType,
                                            typename AimInputType::DeviceType>;
        using AimInputShapeType = RemConstRef<decltype(std::declval<AimInputType>().Shape())>;
        using AimGradType = typename GradMap::template Find<LayerOutput>;
        
    public:
        BiasLayer(std::string p_name, Shape<ParamCategory> p_shape)
            : m_name(std::move(p_name))
            , m_biasShape(std::move(p_shape))
        { }
        
        template <typename... TShapeParams>
        BiasLayer(std::string p_name, size_t val, TShapeParams&&... shapeParams)
            : m_name(std::move(p_name))
            , m_biasShape(val, std::forward<TShapeParams>(shapeParams)...)
        { }
        
        template <typename TInitializer, typename TBuffer, 
                  typename TInitPolicies = typename TInitializer::PolicyCont>
        void Init(TInitializer& initializer, TBuffer& loadBuffer)
        {
            if (auto matPtr = loadBuffer.template TryGet<ParamCategory>(m_name); matPtr)
            {
                if (matPtr->Shape() != m_biasShape)
                {
                    throw std::runtime_error("Load parameter error in BiasLayer");
                }
                m_bias = *matPtr;
                return;
            }
            else if (initializer.template IsParamExist<ParamCategory>(m_name))
            {
                m_bias = ParamType(m_biasShape);
                initializer.GetParam(m_name, m_bias);
                loadBuffer.Set(m_name, m_bias);
                return;
            }
            else
            {
                m_bias = ParamType(m_biasShape);
                using CurInitializer = PickInitializer<TInitPolicies, InitPolicy::BiasTypeCate>;
                if constexpr (!std::is_same<CurInitializer, void>::value)
                {
                    auto& cur_init = initializer.template GetFiller<CurInitializer>();
                    cur_init.Fill(m_bias, m_biasShape.Count(), m_biasShape.Count());
                    loadBuffer.Set(m_name, m_bias);
                }
                else
                {
                    throw std::runtime_error("Cannot get initializer for InitPolicy::BiasTypeCate");
                }
            }
        }
        
        template <typename TSave>
        void SaveWeights(TSave& saver) const
        {
            auto matPtr = saver.template TryGet<ParamCategory>(m_name);
            if (matPtr && (*matPtr != m_bias))
            {
                throw std::runtime_error("Duplicate save for data: " + m_name);
            }
            saver.Set(m_name, m_bias);
        }
        
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto input = LayerTraits::PickItemFromCont<InputMap, LayerInput>(std::forward<TIn>(p_in));
            if constexpr (IsFeedbackOutput)
            {
                m_inputShapeStack.push(input.Shape());
            }
            
            return LayerOutputCont<BiasLayer>().template Set<LayerOutput>(input + Duplicate(m_bias, input.Shape()));
        }
        
        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (IsUpdate)
            {
                auto grad = LayerTraits::PickItemFromCont<GradMap, LayerOutput>(p_grad);
                m_paramGradStack.push(Collapse(std::move(grad), m_biasShape));
            }
            
            if constexpr (IsFeedbackOutput)
            {
                auto grad = LayerTraits::PickItemFromCont<GradMap, LayerOutput>(std::forward<TGrad>(p_grad));
                
                if (m_inputShapeStack.empty())
                {
                    throw std::runtime_error("Cannot feed back in BiasLayer");
                }
                auto curShape = m_inputShapeStack.top();
                m_inputShapeStack.pop();

                return LayerInputCont<BiasLayer>().template Set<LayerInput>(Collapse(std::move(grad), curShape));
            }
            else
                return LayerInputCont<BiasLayer>();
        }
        
        template <typename TGradCollector>
        void GradCollect(TGradCollector& col)
        {
            if constexpr (IsUpdate)
            {
                LayerTraits::MatrixGradCollect(m_bias, m_paramGradStack, col);
            }
        }

        void NeutralInvariant() const
        {
            if constexpr (IsUpdate)
            {
                if (!m_paramGradStack.empty())
                {
                    throw std::runtime_error("NeutralInvariant Fail!");
                }
            }
            if constexpr (IsFeedbackOutput)
            {
                if (!m_inputShapeStack.empty())
                {
                    throw std::runtime_error("NeutralInvariant Fail!");
                }
            }
        }
        
    private:
        std::string m_name;
        Shape<ParamCategory> m_biasShape;
        ParamType m_bias;
        
        typename NSBiasLayer::GradStackType_<AimGradType, Shape<ParamCategory>, IsUpdate>::type m_paramGradStack;
        LayerTraits::LayerInternalBuf<AimInputShapeType, IsFeedbackOutput> m_inputShapeStack;
    };
}