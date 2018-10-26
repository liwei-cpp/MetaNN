#pragma once

#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/data/cardinal/matrix/matrix_base.h>
#include <MetaNN/evaluate/facilities/eval_buffer.h>
#include <MetaNN/evaluate/facilities/eval_group.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <cassert>
#include <stdexcept>

namespace MetaNN
{
namespace NSZeroMatrix
{
template <typename TElem, typename TDevice>
class EvalUnit;

template <typename TElement>
class EvalUnit<TElement, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    EvalUnit(EvalHandle<Matrix<TElement, DeviceTags::CPU>> resBuf,
             size_t rowNum, size_t colNum)
        : m_resHandle(std::move(resBuf))
        , m_rowNum(rowNum)
        , m_colNum(colNum) {}

    void Eval() override
    {
        m_resHandle.Allocate(m_rowNum, m_colNum);
        auto lowLayer = LowerAccess(m_resHandle.MutableData());
        auto mem = lowLayer.MutableRawMemory();
        
        memset(mem, 0, sizeof(TElement) * m_colNum * m_rowNum);
        m_resHandle.SetEval();
    }

private:
    EvalHandle<Matrix<TElement, DeviceTags::CPU>> m_resHandle;
    size_t m_rowNum;
    size_t m_colNum;
};
}

template <typename TElem, typename TDevice>
class ZeroMatrix : public Shape<CategoryTags::Matrix>
{
public:
    using ElementType = TElem;
    using DeviceType = TDevice;
    
public:
    ZeroMatrix(size_t p_rowNum, size_t p_colNum)
        : Shape<CategoryTags::Matrix>(p_rowNum, p_colNum)
    {}

    bool operator== (const ZeroMatrix& val) const
    {
        return (GetShape() == val.GetShape());
    }

    template <typename TOtherType,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<TOtherType>, ZeroMatrix>>>
    bool operator== (const TOtherType&) const
    {
        return false;
    }

    template <typename TData>
    bool operator!= (const TData& val) const
    {
        return !(operator==(val));
    }

    auto EvalRegister() const
    {
        using TEvalUnit = NSZeroMatrix::EvalUnit<ElementType, DeviceType>;
        using TEvalGroup = TrivalEvalGroup<TEvalUnit>;
        if (!m_evalBuf.IsEvaluated())
        {
            auto evalHandle = m_evalBuf.Handle();
            decltype(auto) outPtr = evalHandle.DataPtr();
            TEvalUnit unit(std::move(evalHandle), RowNum(), ColNum());
            EvalPlan<DeviceType>::template Register<TEvalGroup>(std::move(unit), outPtr, {});
        }
        return m_evalBuf.ConstHandle();
    }
    
private:
    EvalBuffer<Matrix<ElementType, DeviceType>> m_evalBuf;
};

template <typename TElem, typename TDevice>
struct DataCategory_<ZeroMatrix<TElem, TDevice>>
{
    using type = CategoryTags::Matrix;
};
}