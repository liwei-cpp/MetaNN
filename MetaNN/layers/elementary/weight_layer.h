#pragma once

namespace MetaNN
{
/*    namespace NSWeightLayer
    {
        template <typename TAimGrad, bool IsUpdate>
        struct GradStackType_
        {
            using type = NullParameter;
        };
        
        template <typename TAimGrad>
        struct GradStackType_<TAimGrad, true>
        {
            using TParamGradStackItem = decltype(Collapse(std::declval<TAimGrad>(), std::declval<Shape<CategoryTags::Matrix>>()));
            using type = LayerTraits::LayerInternalBuf<TParamGradStackItem, true>;
        };
    }

    struct LayerInput;
    struct LayerOutput;

    template <typename TInputs, typename TGrads, typename TPolicies>
    class WeightLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurLayerPolicy = PlainPolicy<TPolicies>;
    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurLayerPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = PolicySelect<GradPolicy, CurLayerPolicy>::IsUpdate;
        
        using InputMap = TInputs;
        using GradMap = FillGradMap<TGrads, LayerOutput>;

    private:
        using AimInputType = typename InputMap::template Find<LayerInput>;
        using ParamType = PrincipalDataType<CategoryTags::Matrix,
                                            typename AimInputType::ElementType,
                                            typename AimInputType::DeviceType>;

    private:
        using AimGradType = typename GradMap::template Find<LayerOutput>;
        auto CalParamGrad(AimGradType&& grad)
        {
            if constexpr (!IsUpdate)
            {
                return NullParameter{};
            }
            else
            {
                if (m_inputStack.empty())
                {
                    throw std::runtime_error("Input stack is empty for WeightLayer Backward.");
                }
                auto tmp = Transpose(m_inputStack.top());
                m_inputStack.pop();
                auto res = Dot(std::move(tmp), std::move(grad));
                return Collapse(std::move(res), m_shape);
            }
        }
        
    public:
        WeightLayer(std::string p_name, Shape<CategoryTags::Matrix> p_shape)
            : m_name(std::move(p_name))
            , m_shape(std::move(p_shape))
        { }
        
        WeightLayer(std::string p_name, size_t rowNum, size_t colNum)
            : m_name(std::move(p_name))
            , m_shape(rowNum, colNum)
        { }

        template <typename TInitializer, typename TBuffer, 
                  typename TInitPolicies = typename TInitializer::PolicyCont>
        void Init(TInitializer& initializer, TBuffer& loadBuffer)
        {
            if (auto matPtr = loadBuffer.template TryGet<CategoryTags::Matrix>(m_name); matPtr)
            {
                if (matPtr->Shape() != m_shape)
                {
                    throw std::runtime_error("Load parameter error in BiasLayer");
                }
                m_weight = *matPtr;
                return;
            }
            else if (initializer.template IsParamExist<CategoryTags::Matrix>(m_name))
            {
                m_weight = ParamType(m_shape);
                initializer.GetParam(m_name, m_weight);
                loadBuffer.Set(m_name, m_weight);
                return;
            }
            else
            {
                m_weight = ParamType(m_shape);
                using CurInitializer = PickInitializer<TInitPolicies, InitPolicy::WeightTypeCate>;
                if constexpr (!std::is_same<CurInitializer, void>::value)
                {
                    auto& cur_init = initializer.template GetFiller<CurInitializer>();
                    cur_init.Fill(m_weight, m_shape.RowNum(), m_shape.ColNum());
                    loadBuffer.Set(m_name, m_weight);
                }
                else
                {
                    throw std::runtime_error("Cannot get initializer for InitPolicy::WeightTypeCate");
                }
            }
        }

        template <typename TSave>
        void SaveWeights(TSave& saver) const
        {
            auto matPtr = saver.template TryGet<CategoryTags::Matrix>(m_name);
            if (matPtr && (*matPtr != m_weight))
            {
                throw std::runtime_error("Duplicate save for matrix: " + m_name);
            }
            saver.Set(m_name, m_weight);
        }

        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            auto input = LayerTraits::PickItemFromCont<InputMap, LayerInput>(std::forward<TIn>(p_in));
            if constexpr (IsUpdate)
            {
                m_inputStack.push(input);
            }
            auto tmpShape = input.Shape();
            tmpShape.RowNum() = m_weight.Shape().RowNum();
            tmpShape.ColNum() = m_weight.Shape().ColNum();
            auto res = Dot(input, Duplicate(m_weight, tmpShape));
            return LayerOutputCont<WeightLayer>().template Set<LayerOutput>(std::move(res));
        }

        template <typename TGrad>
        auto FeedBackward(TGrad&& p_grad)
        {
            if constexpr (IsUpdate)
            {
                auto grad = LayerTraits::PickItemFromCont<GradMap, LayerOutput>(p_grad);
                m_paramGradStack.push(CalParamGrad(std::move(grad)));
            }

            if constexpr (IsFeedbackOutput)
            {
                auto grad = LayerTraits::PickItemFromCont<GradMap, LayerOutput>(std::forward<TGrad>(p_grad));
                auto tmpShape = grad.Shape();
                tmpShape.ColNum() = m_weight.Shape().RowNum();
                tmpShape.RowNum() = m_weight.Shape().ColNum();
                auto res = Dot(std::move(grad), Duplicate(Transpose(m_weight), tmpShape));

                return LayerInputCont<WeightLayer>().template Set<LayerInput>(std::move(res));
            }
            else
                return LayerInputCont<WeightLayer>();
        }

        template <typename TGradCollector>
        void GradCollect(TGradCollector& col)
        {
            if constexpr (IsUpdate)
            {
                LayerTraits::ParamGradCollect(m_weight, m_paramGradStack, col);
            }
        }

        void NeutralInvariant() const
        {
            if constexpr (IsUpdate)
            {
                if ((!m_paramGradStack.empty()) || (!m_inputStack.empty()))
                {
                    throw std::runtime_error("NeutralInvariant Fail!");
                }
            }
        }
    private:
        std::string m_name;
        Shape<CategoryTags::Matrix> m_shape;
        ParamType m_weight;
        
        LayerTraits::LayerInternalBuf<AimInputType, IsUpdate> m_inputStack;
        std::stack<decltype(std::declval<WeightLayer>().CalParamGrad(std::declval<AimGradType>()))> m_paramGradStack;
    };*/
}