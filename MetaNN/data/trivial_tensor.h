#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_buffer.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <MetaNN/facilities/_.h>
#include <type_traits>

namespace MetaNN
{
    namespace NSTrivialTensor
    {
        template <typename TScalarHandle, typename TOutputHandle, size_t uDim>
        class EvalItem : public BaseEvalItem
        {
        public:
            using DeviceType = DeviceTypeFromHandle<TOutputHandle>;

            EvalItem(TScalarHandle p_scalar, TOutputHandle resBuf,
                     Shape<uDim> p_shape)
                : BaseEvalItem(TypeID<EvalItem>(),
                               { p_scalar.DataPtr() }, resBuf.DataPtr())
                , m_resHandle(std::move(resBuf))
                , m_shape(std::move(p_shape))
                , m_scalarHandle(std::move(p_scalar))
            { }
        
            TOutputHandle m_resHandle;
            Shape<uDim> m_shape;
            TScalarHandle m_scalarHandle;
        };
    
        template <typename TScalarHandle, typename TOutputHandle, size_t uDim>
        class EvalGroup : public TrivialEvalGroup<EvalItem<TScalarHandle, TOutputHandle, uDim>>
        {
            using EvalItemType = EvalItem<TScalarHandle, TOutputHandle, uDim>;

        protected:
            virtual void EvalInternalLogic(EvalItemType& evalItem) final override
            {
                const auto& in = evalItem.m_scalarHandle.Data();

                using ResType = typename TOutputHandle::DataType;
                using ElementType = typename ResType::ElementType;

                static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>,
                              "Currently only CPU is supported.");

                ResType out(evalItem.m_shape);
                auto lowLayer = LowerAccess(out);
                auto mem = lowLayer.MutableRawMemory();

                const size_t elemCount = evalItem.m_shape.Count();
                const ElementType val = static_cast<ElementType>(in.Value());
                for (size_t i = 0; i < elemCount; ++i)
                {
                    mem[i] = val;
                }
                evalItem.m_resHandle.SetData(std::move(out));
            }
        };
    }

    template<typename TScalar, size_t uDim>
    class TrivialTensor
    {
    public:
        using CategoryTag = CategoryTags::Tensor<uDim>;
        using ElementType = typename TScalar::ElementType;
        using DeviceType = typename TScalar::DeviceType;

    public:
        template <typename...TParams>
        explicit TrivialTensor(TScalar p_scalar, TParams&&... params)
            : m_shape(std::forward<TParams>(params)...)
            , m_scalar(std::move(p_scalar))
        {}

        const auto& Shape() const noexcept
        {
            return m_shape;
        }

        bool operator== (const TrivialTensor& val) const
        {
            return (m_shape == val.m_shape) &&
                   (m_scalar == val.m_scalar);
        }

        auto EvalRegister() const
        {
            if (!m_evalBuf.IsEvaluated())
            {
                auto outHandle = m_evalBuf.Handle();
        
                if (!EvalPlan::Inst().IsAlreadyRegisted(outHandle.DataPtr()))
                {
                    auto handle = m_scalar.EvalRegister();

                    using ItemType = NSTrivialTensor::EvalItem<decltype(handle), decltype(outHandle), uDim>;
                    using GroupType = NSTrivialTensor::EvalGroup<decltype(handle), decltype(outHandle), uDim>;
                    using DispatcherType = TrivialEvalItemDispatcher<GroupType>;

                    auto item = std::make_unique<ItemType>(std::move(handle), std::move(outHandle), m_shape);
                    EvalPlan::Inst().Register<DispatcherType>(std::move(item));
                }
            }
            return m_evalBuf.ConstHandle();
        }

        const auto& Scalar() const
        {
            return m_scalar;
        }

    private:
        MetaNN::Shape<uDim> m_shape;
        TScalar  m_scalar;
        EvalBuffer<Tensor<ElementType, DeviceType, uDim>> m_evalBuf;
    };
    
    template<typename TScalar, typename... TShapeParams>
    TrivialTensor(TScalar, TShapeParams&&...) -> TrivialTensor<TScalar, sizeof...(TShapeParams)>;
}