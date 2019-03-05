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
    namespace NSZeroScalar
    {
    template <typename TElement, typename TDevice>
    class EvalUnit : public BaseEvalUnit<TDevice>
    {
    public:
        EvalUnit(EvalHandle<Scalar<TElement, TDevice>> resBuf)
            : m_resHandle(std::move(resBuf))
        {}

        void Eval() override
        {
            m_resHandle.Allocate();
            auto lowLayer = LowerAccess(m_resHandle.MutableData());
            auto mem = lowLayer.MutableRawMemory();
        
            static_assert(std::is_same_v<TDevice, DeviceTags::CPU>, 
                          "Memset not support for other device tag.");
            *mem = TElement{};
            m_resHandle.SetEval();
        }
    private:
        EvalHandle<Scalar<TElement, TDevice>> m_resHandle;
    };
}

    template <typename TElem, typename TDevice>
    class ZeroScalar
    {
    public:
        using CategoryTag = CategoryTags::Scalar;
        using ElementType = TElem;
        using DeviceType = TDevice;

    public:
        explicit ZeroScalar(MetaNN::Shape<CategoryTag> p_shape = MetaNN::Shape<CategoryTag>())
        {}

        const auto& Shape() const noexcept
        {
            const static MetaNN::Shape<CategoryTag> shape;
            return shape;
        }

        bool operator== (const ZeroScalar&) const noexcept
        {
            return true;
        }

        auto EvalRegister() const
        {
            using TEvalUnit = NSZeroScalar::EvalUnit<ElementType, DeviceType>;
            using TEvalGroup = TrivalEvalGroup<TEvalUnit>;
            if (!m_evalBuf.IsEvaluated())
            {
                auto evalHandle = m_evalBuf.Handle();
                decltype(auto) outPtr = evalHandle.DataPtr();
                TEvalUnit unit(std::move(evalHandle));
                EvalPlan<DeviceType>::template Register<TEvalGroup>(std::move(unit), outPtr, {});
            }
            return m_evalBuf.ConstHandle();
        }
    
    private:
        EvalBuffer<Scalar<ElementType, DeviceType>> m_evalBuf;
    };
}