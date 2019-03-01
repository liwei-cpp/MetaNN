#pragma once

#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>

namespace MetaNN
{
    struct ParamPolicy
    {
        using MajorClass = ParamPolicy;
        
        struct ParamDeviceTypeCate
        {
            using CPU = DeviceTags::CPU;
        };
        using ParamDevice = ParamDeviceTypeCate::CPU;
        
        struct ParamTypeTypeCate;
        using ParamType = float;

        struct ParamCategoryTypeCate
        {
            using Scalar = CategoryTags::Scalar;
            using Matrix = CategoryTags::Matrix;
            using ThreeDArray = CategoryTags::ThreeDArray;
        };
        using ParamCategory = ParamCategoryTypeCate::Matrix;
        
        struct InitializerTypeCate;
        using Initializer = NullParameter;
    };
#include <MetaNN/policies/policy_macro_begin.h>
    TypePolicyObj(PScalarParam,               ParamPolicy, ParamCategory, Scalar);
    TypePolicyObj(PMatrixParam,               ParamPolicy, ParamCategory, Matrix);
    TypePolicyObj(PThreeDArrayParam,          ParamPolicy, ParamCategory, ThreeDArray);
    
    TypePolicyTemplate(PParamElementTypeIs,   ParamPolicy, ParamType);

    TypePolicyObj(PCPUDeviceParam,            ParamPolicy, ParamDevice,   CPU);
    
    TypePolicyTemplate(PInitializerIs,        ParamPolicy, Initializer);
#include <MetaNN/policies/policy_macro_end.h>

    struct LayerOutput;

    template <typename TInputs, typename TGrads, typename TPolicies>
    class ParamSourceLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        static constexpr bool IsFeedbackOutput = false;
        static constexpr bool IsUpdate = PolicySelect<GradPolicy, CurLayerPolicy>::IsUpdate;

    private:
        using ParamCategory = typename PolicySelect<ParamPolicy, CurLayerPolicy>::ParamCategory;
        using ElementType = typename PolicySelect<ParamPolicy, CurLayerPolicy>::ParamType;
        using DeviceType  = typename PolicySelect<ParamPolicy, CurLayerPolicy>::ParamDevice;
        using ParamType = PrincipalDataType<ParamCategory, ElementType, DeviceType>;

    public:
        using InputMap = LayerIOMap<>;
        using GradMap = FillGradMap<TGrads, LayerOutput>;

    public:
        ParamSourceLayer(std::string name)
            : m_name(std::move(name))
        {}

        template <typename TInitializer, typename TBuffer>
        void Init(TInitializer& initializer, TBuffer& loadBuffer)
        {
            if (auto matPtr = loadBuffer.template TryGet<ParamCategory>(m_name); matPtr)
            {
                if (matPtr->Shape() != m_dataShape)
                {
                    throw std::runtime_error("Load parameter error: shape mismatch");
                }
                m_data = *matPtr;
                return;
            }
            
            m_data = ParamType(m_dataShape);
            if (initializer.template IsParamExist<ParamCategory>(m_name))
            {
                initializer.GetParam(m_name, m_data);
            }
            else
            {
                m_data = ParamType(m_dataShape);
                using InitializerName = typename PolicySelect<ParamPolicy, CurLayerPolicy>::Initializer;
                auto& cur_init = initializer.template GetFiller<InitializerName>();
                cur_init.Fill(m_data);
            }
            loadBuffer.Set(m_name, m_data);
        }
        
        template <typename TSave>
        void SaveWeights(TSave& saver) const
        {
            auto matPtr = saver.template TryGet<ParamCategory>(m_name);
            if (matPtr && (*matPtr != m_data))
            {
                throw std::runtime_error("Duplicate save for data: " + m_name);
            }
            saver.Set(m_name, m_data);
        }
        
        template <typename TGradCollector>
        void GradCollect(TGradCollector& col)
        {
            if constexpr (IsUpdate)
            {
                LayerTraits::ParamGradCollect(m_data, m_paramGradStack, col);
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
        }
        
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            return LayerOutputCont<ParamSourceLayer>().template Set<LayerOutput>(m_data);
        }

        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (IsUpdate)
            {
                auto grad = LayerTraits::PickItemFromCont<GradMap, LayerOutput>(p_grad);
                if (grad.Shape() != m_data.Shape())
                {
                    throw std::runtime_error("Parameter and its grad shape mismatch.");
                }
                m_paramGradStack.push(grad);
            }
            return LayerInputCont<ParamSourceLayer>();
        }

    private:
        std::string m_name;
        Shape<ParamCategory> m_dataShape;
        ParamType m_data;
        
        using AimGradType = typename GradMap::template Find<LayerOutput>;
        LayerTraits::LayerInternalBuf<AimGradType, IsUpdate> m_paramGradStack;
    };
}