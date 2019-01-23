#pragma once

namespace MetaNN
{
    template <typename TInputItems, typename TInputGrads, typename TPolicies>
    class BiasLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;
    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = PolicySelect<GradPolicy, CurLayerPolicy>::IsUpdate;
        
        using InputContType = LayerIO;
        using OutputContType = LayerIO;
        
        using InputItemTypes = TInputItems;
        using InputGradTypes = TInputGrads;

    private:
        using ParamCategory = PolicySelect<ParamPolicy, CurLayerPolicy>::ParamType;
        using AimInputType = typename TInputMap::template Find<LayerIO>;
        using ParamType = PrincipalDataType<ParamCategory,
                                            typename AimInputType::ElementType,
                                            typename AimInputType::DeviceType>;
        using AimInputShapeType = RemConstRef<decltype(std::declval<AimInputType>().Shape())>;
        using AimGradType = typename InputGradTypes::template Find<LayerIO>;
        
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
        void Init(TInitializer& initializer, TBuffer& loadBuffer, std::ostream* log = nullptr)
        {
            if (auto matPtr = loadBuffer.TryGet<ParamCategory>(m_name); matPtr)
            {
                if (matPtr->Shape() != m_biasShape)
                {
                    throw std::runtime_error("Load parameter error in BiasLayer");
                }
                m_bias = *matPtr;
                if (log)
                {
                    std::string logInfo = "Load from load buffer: " + m_name + '\n';
                    (*log) << logInfo;
                }
                return;
            }
            else if (initializer.IsParamExist<ParamCategory>(m_name))
            {
                m_bias = ParamType(m_biasShape);
                initializer.CopyParam<ParamCategory>(m_name, m_bias);
                loadBuffer.Set<ParamCategory>(m_name, m_bias);
                if (log)
                {
                    std::string logInfo = "Copy from initializer: " + m_name + '\n';
                    (*log) << logInfo;
                }
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
                    loadBuffer.Set<ParamCategory>(m_name, m_bias);
                    if (log)
                    {
                        std::string logInfo = "Random init from initializer: " + m_name + '\n';
                        (*log) << logInfo;
                    }
            }
            else
            {
                throw std::runtime_error("Cannot get initializer for InitPolicy::BiasTypeCate");
            }
        }
        
        template <typename TSave>
        void SaveWeights(TSave& saver) const
        {
            auto matPtr = saver.TryGet<ParamCategory>(m_name);
            if (matPtr && (matPtr != m_bias))
            {
                throw std::runtime_error("Duplicate save for matrix: " + m_name);
            }
            loadBuffer.Set<ParamCategory>(m_name, m_bias);
        }
        
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto input = LayerTraits::PickItemFromCont<InputItemTypes, LayerIO>(std::forward<TIn>(p_in));
            
            if constexpr (IsFeedbackOutput)
            {
                m_inputShapeStack.push(input.Shape());
            }
            
            auto proShape = LayerTraits::ShapePromote(input.Shape(), m_biasShape);
            return OutputContType::Create().template Set<LayerIO>(Duplicate(input, proShape) + Duplicate(m_bias, proShape));
        }
        
        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (IsUpdate)
            {
                auto grad = LayerTraits::PickItemFromCont<InputGradTypes, LayerIO>(p_grad);
                m_paramGradStack.push(Collapse(std::move(grad), m_biasShape));
            }
            
            if constexpr (IsFeedbackOutput)
            {
                auto grad = LayerTraits::PickItemFromCont<InputGradTypes, LayerIO>(std::forward<TGrad>(p_grad));
                
                if (m_inputShapeStack.empty())
                {
                    throw std::runtime_error("Cannot feed back in BiasLayer");
                }
                auto curShape = m_inputShapeStack.top();
                m_inputShapeStack.pop();

                return LayerIO::Create().template Set<LayerIO>(Collapse(std::move(grad), curShape));
            }
            else
                return LayerIO::Create();
        }
        
        template <typename TGradCollector>
        void GradCollect(TGradCollector& col)
        {
            if constexpr (IsUpdate)
            {
                LayerTraits::MatrixGradCollect(m_bias, m_grad, col);
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
    }
        
    private:
        std::string m_name;
        Shape<ParamCategory> m_biasShape;
        ParamType m_bias;
        
        using TParamGradStackItem = decltype(Collapse(std::declval<AimGradType>(), m_biasShape));
        
        LayerTraits::LayerInternalBuf<TParamGradStackItem, IsUpdate> m_paramGradStack;
        LayerTraits::LayerInternalBuf<AimInputShapeType, IsFeedbackOutput> m_inputShapeStack;
    };
}