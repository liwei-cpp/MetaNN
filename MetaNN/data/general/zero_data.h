#pragma once

#include <cstring>
#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/data/cardinal/matrix/matrix.h>
#include <MetaNN/evaluate/facilities/eval_buffer.h>
#include <MetaNN/evaluate/facilities/eval_group.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>

namespace MetaNN
{
    namespace NSZeroData
    {
        template <typename TCategory, typename TElement, typename TDevice>
        class EvalUnit : public BaseEvalUnit<TDevice>
        {
        public:
            EvalUnit(EvalHandle<PrincipalDataType<TCategory, TElement, TDevice>> resBuf,
                     Shape<TCategory> p_shape)
                : m_resHandle(std::move(resBuf))
                , m_shape(std::move(p_shape))
            {}

            void Eval() override
            {
                PrincipalDataType<TCategory, TElement, TDevice> res(m_shape);
                auto lowLayer = LowerAccess(res);
                auto mem = lowLayer.MutableRawMemory();
        
                static_assert(std::is_same_v<TDevice, DeviceTags::CPU>, 
                              "Memset not support for other device tag.");
                memset(mem, 0, sizeof(TElement) * m_shape.Count());
                m_resHandle.SetData(std::move(res));
            }

        private:
            EvalHandle<PrincipalDataType<TCategory, TElement, TDevice>> m_resHandle;
            const Shape<TCategory> m_shape;
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
            using TEvalUnit = NSZeroData::EvalUnit<CategoryTag, ElementType, DeviceType>;
            using TEvalGroup = TrivalEvalGroup<TEvalUnit>;
            if (!m_evalBuf.IsEvaluated())
            {
                auto evalHandle = m_evalBuf.Handle();
                decltype(auto) outPtr = evalHandle.DataPtr();
                TEvalUnit unit(std::move(evalHandle), m_shape);
                EvalPlan<DeviceType>::template Register<TEvalGroup>(std::move(unit), outPtr, {});
            }
            return m_evalBuf.ConstHandle();
        }

    private:
        MetaNN::Shape<CategoryTag> m_shape;
        EvalBuffer<PrincipalDataType<CategoryTag, ElementType, DeviceType>> m_evalBuf;
    };
}