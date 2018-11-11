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
namespace NSZeroMatrix
{
template <typename TElement, typename TDevice>
class EvalUnit : public BaseEvalUnit<TDevice>
{
public:
    EvalUnit(EvalHandle<Matrix<TElement, TDevice>> resBuf,
             Shape<CategoryTags::Matrix> p_shape)
        : m_resHandle(std::move(resBuf))
        , m_shape(std::move(p_shape))
    {}

    void Eval() override
    {
        m_resHandle.Allocate(m_shape);
        auto lowLayer = LowerAccess(m_resHandle.MutableData());
        auto mem = lowLayer.MutableRawMemory();
        
        static_assert(std::is_same_v<TDevice, DeviceTags::CPU>, 
                      "Memset not support for other device tag.");
        memset(mem, 0, sizeof(TElement) * m_shape.Count());
        m_resHandle.SetEval();
    }

private:
    EvalHandle<Matrix<TElement, TDevice>> m_resHandle;
    const Shape<CategoryTags::Matrix> m_shape;
};
}

template <typename TElem, typename TDevice>
class ZeroMatrix
{
public:
    using CategoryTag = CategoryTags::Matrix;
    using ElementType = TElem;
    using DeviceType = TDevice;

public:
    explicit ZeroMatrix(size_t rowNum, size_t colNum)
        : m_shape(rowNum, colNum)
    {}
    
    explicit ZeroMatrix(MetaNN::Shape<CategoryTag> p_shape = MetaNN::Shape<CategoryTag>())
        : m_shape(std::move(p_shape))
    {}
    
    const auto& Shape() const noexcept
    {
        return m_shape;
    }

    bool operator== (const ZeroMatrix& val) const
    {
        return (m_shape == val.m_shape);
    }

    auto EvalRegister() const
    {
        using TEvalUnit = NSZeroMatrix::EvalUnit<ElementType, DeviceType>;
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
    EvalBuffer<Matrix<ElementType, DeviceType>> m_evalBuf;
};
}
