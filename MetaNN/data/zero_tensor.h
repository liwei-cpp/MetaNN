#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_buffer.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <MetaNN/facilities/_.h>
#include <cstring>
#include <type_traits>

namespace MetaNN
{
    namespace NSZeroTensor
    {
        template <typename TElem, typename TDevice, size_t uDim>
        class EvalItem : public BaseEvalItem
        {
        public:
            using CategoryTag = CategoryTags::Tensor<uDim>;
            using ElementType = TElem;
            using DeviceType = TDevice;

            EvalItem(EvalHandle<PrincipalDataType<CategoryTag, ElementType, DeviceType>> resBuf,
                     Shape<uDim> p_shape)
                : BaseEvalItem(TypeID<EvalItem>(), {}, resBuf.DataPtr())
                , m_resHandle(std::move(resBuf))
                , m_shape(std::move(p_shape))
            {
            }
        
            EvalHandle<PrincipalDataType<CategoryTag, ElementType, DeviceType>> m_resHandle;
            const Shape<uDim> m_shape;
        };

        template <typename TElem, typename TDevice, size_t uDim>
        class EvalGroup : public TrivialEvalGroup<EvalItem<TElem, TDevice, uDim>>
        {
            using EvalItemType = EvalItem<TElem, TDevice, uDim>;
        protected:
            virtual void EvalInternalLogic(EvalItemType& evalItem) final override
            {
                using CategoryTag = CategoryTags::Tensor<uDim>;
                PrincipalDataType<CategoryTag, TElem, TDevice> res(evalItem.m_shape);
                static_assert(std::is_same_v<TDevice, DeviceTags::CPU>, 
                              "Only CPU is supported now.");

                if constexpr (uDim == 0)
                {
                    res.SetValue(0);
                }
                else
                {
                    auto lowLayer = LowerAccess(res);
                    auto mem = lowLayer.MutableRawMemory();

                    const unsigned bufLen = static_cast<unsigned>(sizeof(TElem) * evalItem.m_shape.Count());
                    assert(bufLen == sizeof(TElem) * evalItem.m_shape.Count());
                    memset(mem, 0, bufLen);
                }
                evalItem.m_resHandle.SetData(std::move(res));
            }
        };
    }

    template <typename TElem, typename TDevice, size_t uDim>
    class ZeroTensor
    {
    public:
        using CategoryTag = CategoryTags::Tensor<uDim>;
        using ElementType = TElem;
        using DeviceType = TDevice;

    public:
        template <typename...TShapeParams,
                  std::enable_if_t<(std::is_convertible_v<TShapeParams, size_t> && ...)>* = nullptr>
        explicit ZeroTensor(TShapeParams&&... shapeParams)
            : m_shape(std::forward<TShapeParams>(shapeParams)...)
        {}
        
        explicit ZeroTensor(MetaNN::Shape<uDim> p_shape)
            : m_shape(std::move(p_shape))
        {}
    
        const auto& Shape() const noexcept
        {
            return m_shape;
        }

        bool operator== (const ZeroTensor& val) const
        {
            return (m_shape == val.m_shape);
        }

        auto EvalRegister() const
        {
            using TEvalItem = NSZeroTensor::EvalItem<ElementType, DeviceType, uDim>;
            using TEvalGroup = NSZeroTensor::EvalGroup<ElementType, DeviceType, uDim>;
            using TItemDispatcher = TrivialEvalItemDispatcher<TEvalGroup>;

            if (!m_evalBuf.IsEvaluated())
            {
                auto evalHandle = m_evalBuf.Handle();
                if (!EvalPlan::Inst().IsAlreadyRegisted(evalHandle.DataPtr()))
                {
                    EvalPlan::Inst().Register<TItemDispatcher>(
                        std::make_unique<TEvalItem>(std::move(evalHandle), m_shape));
                }
            }
            return m_evalBuf.ConstHandle();
        }

    private:
        MetaNN::Shape<uDim> m_shape;
        EvalBuffer<PrincipalDataType<CategoryTag, ElementType, DeviceType>> m_evalBuf;
    };
}