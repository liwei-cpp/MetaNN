#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_buffer.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <type_traits>

namespace MetaNN
{
    namespace NSTrivalTensor
    {
        template <typename TScalar, size_t uDim>
        class EvalItem : public BaseEvalItem<typename TScalar::DeviceType>
        {
        public:
            using ElementType = typename TScalar::ElementType;
            using DeviceType = typename TScalar::DeviceType;

            EvalItem(EvalHandle<Tensor<ElementType, DeviceType, uDim>> resBuf,
                     Shape<uDim> p_shape, TScalar p_scalar)
                : BaseEvalItem<DeviceType>(std::type_index(typeid(EvalItem)),
                                           {}, resBuf.DataPtr())
                , m_resHandle(std::move(resBuf))
                , m_shape(std::move(p_shape))
                , m_scalar(std::move(p_scalar))
            { }
        
            EvalHandle<Tensor<ElementType, DeviceType, uDim>> m_resHandle;
            Shape<uDim> m_shape;
            TScalar  m_scalar;
        };
    
        template <typename TScalar, size_t uDim>
        class EvalGroup : public TrivalEvalGroup<EvalItem<TScalar, uDim>>
        {
            using EvalItemType = EvalItem<TScalar, uDim>;

        protected:
            virtual void EvalInternalLogic(EvalItemType& evalItem) final override
            {
                using ElementType = typename TScalar::ElementType;
                using DeviceType = typename TScalar::DeviceType;

                static_assert(std::is_same_v<DeviceType, DeviceTags::CPU>,
                              "Currently only CPU is supported.");

                Tensor<ElementType, DeviceType, uDim> out(evalItem.m_shape);
                auto lowLayer = LowerAccess(out);
                auto mem = lowLayer.MutableRawMemory();

                const size_t elemCount = evalItem.m_shape.Count();
                const ElementType val = static_cast<ElementType>(evalItem.m_scalar.Value());
                for (size_t i = 0; i < elemCount; ++i)
                {
                    mem[i] = val;
                }
                evalItem.m_resHandle.SetData(std::move(out));
            }
        };
    }

    template<typename TScalar, size_t uDim>
    class TrivalTensor
    {
        static_assert(uDim > 0);
    public:
        using CategoryTag = CategoryTags::Tensor<uDim>;
        using ElementType = typename TScalar::ElementType;
        using DeviceType = typename TScalar::DeviceType;

    public:
        template <typename...TParams>
        explicit TrivalTensor(TScalar p_scalar, TParams&&... params)
            : m_shape(std::forward<TParams>(params)...)
            , m_scalar(std::move(p_scalar))
        {}

        const auto& Shape() const noexcept
        {
            return m_shape;
        }

        bool operator== (const TrivalTensor& val) const
        {
            return (m_shape == val.m_shape) &&
                   (m_scalar == val.m_scalar);
        }

        auto EvalRegister() const
        {
            using TEvalItem = NSTrivalTensor::EvalItem<TScalar, uDim>;
            using TEvalGroup = NSTrivalTensor::EvalGroup<TScalar, uDim>;
            using TItemDispatcher = TrivalEvalItemDispatcher<TEvalGroup>;

            if (!m_evalBuf.IsEvaluated())
            {
                auto evalHandle = m_evalBuf.Handle();
                if (!EvalPlan<DeviceType>::Inst().IsAlreayRegisted(evalHandle.DataPtr()))
                {
                    EvalPlan<DeviceType>::Inst().template Register<TItemDispatcher>(
                        std::make_unique<TEvalItem>(std::move(evalHandle), m_shape, m_scalar));
                }
            }
            return m_evalBuf.ConstHandle();
        }

        const auto& ElementValue() const
        {
            return m_scalar;
        }

    private:
        MetaNN::Shape<uDim> m_shape;
        TScalar  m_scalar;
        EvalBuffer<Tensor<ElementType, DeviceType, uDim>> m_evalBuf;
    };
    
    template<typename TScalar, typename... TShapeParams>
    TrivalTensor(TScalar, TShapeParams&&...) -> TrivalTensor<TScalar, sizeof...(TShapeParams)>;
}