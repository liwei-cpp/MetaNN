#pragma once

#include <cstring>
#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/data/cardinal/matrix/matrix.h>
#include <MetaNN/evaluate/eval_buffer.h>
#include <MetaNN/evaluate/eval_plan.h>

namespace MetaNN
{
    namespace NSZeroData
    {
        template <typename TCategory, typename TElement, typename TDevice>
        class EvalItem : public BaseEvalItem<TDevice>
        {
        public:
            EvalItem(EvalHandle<PrincipalDataType<TCategory, TElement, TDevice>> resBuf,
                     Shape<TCategory> p_shape)
                : BaseEvalItem<TDevice>(std::type_index(typeid(EvalItem)),
                                        {}, resBuf.DataPtr())
                , m_resHandle(std::move(resBuf))
                , m_shape(std::move(p_shape))
            {
            }
        
            EvalHandle<PrincipalDataType<TCategory, TElement, TDevice>> m_resHandle;
            const Shape<TCategory> m_shape;
        };

        template <typename TCategory, typename TElement, typename TDevice>
        class EvalGroup : public TrivalEvalGroup<EvalItem<TCategory, TElement, TDevice>>
        {
            using EvalItemType = EvalItem<TCategory, TElement, TDevice>;
        protected:
            virtual void EvalInternalLogic(EvalItemType& evalItem) final override
            {
                PrincipalDataType<TCategory, TElement, TDevice> res(evalItem.m_shape);
                auto lowLayer = LowerAccess(res);
                auto mem = lowLayer.MutableRawMemory();
        
                static_assert(std::is_same_v<TDevice, DeviceTags::CPU>, 
                              "Memset not support for other device tag.");
                memset(mem, 0, sizeof(TElement) * evalItem.m_shape.Count());
                evalItem.m_resHandle.SetData(std::move(res));
            }
        };
    }

    template <typename TCategory, typename TElem, typename TDevice>
    class ZeroData
    {
    public:
        using CategoryTag = TCategory;
        using ElementType = TElem;
        using DeviceType = TDevice;

    public:
        explicit ZeroData(MetaNN::Shape<CategoryTag> p_shape = MetaNN::Shape<CategoryTag>())
            : m_shape(std::move(p_shape))
        {}
        
        template <typename...TShapeParams>
        explicit ZeroData(size_t val, TShapeParams&&... shapeParams)
            : m_shape(val, std::forward<TShapeParams>(shapeParams)...)
        {}
    
        const auto& Shape() const noexcept
        {
            return m_shape;
        }

        bool operator== (const ZeroData& val) const
        {
            return (m_shape == val.m_shape);
        }

        auto EvalRegister() const
        {
            using TEvalItem = NSZeroData::EvalItem<CategoryTag, ElementType, DeviceType>;
            using TEvalGroup = NSZeroData::EvalGroup<CategoryTag, ElementType, DeviceType>;
            using TItemDispatcher = TrivalEvalItemDispatcher<TEvalGroup>;

            if (!m_evalBuf.IsEvaluated())
            {
                auto evalHandle = m_evalBuf.Handle();
                if (!EvalPlan<DeviceType>::Inst().IsAlreayRegisted(evalHandle.DataPtr()))
                {
                    EvalPlan<DeviceType>::Inst().template Register<TItemDispatcher>(
                        std::make_unique<TEvalItem>(std::move(evalHandle), m_shape));
                }
            }
            return m_evalBuf.ConstHandle();
        }

    private:
        MetaNN::Shape<CategoryTag> m_shape;
        EvalBuffer<PrincipalDataType<CategoryTag, ElementType, DeviceType>> m_evalBuf;
    };
}