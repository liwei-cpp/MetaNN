#pragma once

#include <MetaNN/facilities/var_type_dict.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/_.h>

namespace MetaNN
{
    template <typename TInputs, typename TPolicies>
    class ParamSourceLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;

    public:
        using ParamType = typename PolicySelect<ParamPolicy, CurLayerPolicy>::ParamType;
        static_assert(!std::is_same_v<ParamType, NullParameter>, "Use PParamTypeIs<> to set parameter type.");

    private:
        using ParamCategory = typename ParamType::CategoryTag;
        using ElementType = typename ParamType::ElementType;
        using DeviceType  = typename ParamType::DeviceType;
        constexpr static bool IsPrincipal = std::is_same_v<PrincipalDataType<ParamCategory, ElementType, DeviceType>, ParamType>;

    public:
        static constexpr bool IsFeedbackOutput = false;
        static constexpr bool IsUpdate = IsPrincipal && PolicySelect<GradPolicy, CurLayerPolicy>::IsUpdate;

    public:
        using InputPortSet = LayerPortSet<>;
        using OutputPortSet = LayerPortSet<struct LayerOutput>;
        using InputMap = typename EmptyLayerInMap_<InputPortSet>::type;

    public:
        template <typename... TParams>
        ParamSourceLayer(std::string name, TParams&&... p_params)
            : m_name(std::move(name))
        {
            if constexpr (IsPrincipal)
            {
                m_dataShape = Shape<ParamCategory::DimNum>(std::forward<TParams>(p_params)...);
            }
            else
            {
                m_data = ParamType(std::forward<TParams>(p_params)...);
            }
        }

        template <typename TInitializer, typename TBuffer>
        void Init(TInitializer& initializer, TBuffer& loadBuffer)
        {
            if constexpr (IsPrincipal)
            {
                m_paramName = initializer.LayerName2ParamName(m_name);
                if (auto matPtr = loadBuffer.template TryGet<ParamCategory>(m_paramName); matPtr)
                {
                    if (matPtr->Shape() != m_dataShape)
                    {
                        throw std::runtime_error("Load parameter error: shape mismatch");
                    }
                    m_data = *matPtr;
                    return;
                }
            
                m_data = ParamType(m_dataShape);
                if (initializer.template IsParamExist<ParamCategory>(m_paramName))
                {
                    initializer.GetParam(m_paramName, m_data);
                }
                else
                {
                    using InitializerName = typename PolicySelect<ParamPolicy, CurLayerPolicy>::Initializer;
                    if constexpr (!std::is_same_v<InitializerName, NullParameter>)
                    {
                        auto& cur_init = initializer.template GetFiller<InitializerName>();
                        cur_init.Fill(m_data);
                    }
                    else
                    {
                        throw std::runtime_error("Cannot get the initializer for layer: " + m_name + " (" + m_paramName + ")");
                    }
                }
                loadBuffer.Set(m_paramName, m_data);
            }
        }
        
        template <typename TSave>
        void SaveWeights(TSave& saver) const
        {
            if constexpr (IsPrincipal)
            {
                auto matPtr = saver.template TryGet<ParamCategory>(m_paramName);
                if (matPtr && (*matPtr != m_data))
                {
                    throw std::runtime_error("Duplicate save for data: " + m_paramName);
                }
                saver.Set(m_paramName, m_data);
            }
        }
        
        template <typename TGradCollector>
        void GradCollect(TGradCollector& col)
        {
            if constexpr (IsUpdate)
            {
                LayerTraits::ParamGradCollect(m_paramName, m_data, m_paramGradStack, col);
            }
        }
        
        void NeutralInvariant() const
        {
            if constexpr (IsUpdate)
            {
                LayerTraits::CheckStackEmpty(m_paramGradStack);
            }
        }
        
        auto FeedForward(const VarTypeDict<>::Values<>&)
        {
            return LayerOutputCont<ParamSourceLayer>().template Set<LayerOutput>(m_data);
        }

        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (IsUpdate && (!RemConstRef<TGrad>::template IsValueEmpty<LayerOutput>))
            {
                auto grad = std::forward<TGrad>(p_grad).template Get<LayerOutput>();
                if (grad.Shape() != m_data.Shape())
                {
                    throw std::runtime_error("Parameter and its grad shape mismatch.");
                }
                m_paramGradStack.push(MakeDynamic(grad));
            }
            return LayerInputCont<ParamSourceLayer>();
        }

    private:
        std::string m_name;
        std::string m_paramName;
        Shape<ParamCategory::DimNum> m_dataShape;
        ParamType m_data;
        
        using AimGradType = DynamicData<ElementType, DeviceType, ParamCategory>;
        LayerTraits::LayerInternalBuf<AimGradType, IsUpdate> m_paramGradStack;
    };
}