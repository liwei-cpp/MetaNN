#pragma once

#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/data/matrices/matrices.h>
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
        const size_t rowLen = lowLayer.RowLen();
        auto mem = lowLayer.MutableRawMemory();
        if (rowLen != m_colNum)
        {
            throw std::runtime_error("Gap among matrix rows");
        }
        
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
class ZeroMatrix
{
public:
    using ElementType = TElem;
    using DeviceType = TDevice;
    
public:
    ZeroMatrix(size_t p_rowNum, size_t p_colNum)
        : m_rowNum(p_rowNum)
        , m_colNum(p_colNum) {}

    bool operator== (const ZeroMatrix& val) const
    {
        return (m_rowNum == val.m_rowNum) &&
               (m_colNum == val.m_colNum);
    }

    template <typename TOtherType>
    bool operator== (const TOtherType&) const
    {
        return false;
    }

    template <typename TData>
    bool operator!= (const TData& val) const
    {
        return !(operator==(val));
    }

    size_t RowNum() const { return m_rowNum; }

    size_t ColNum() const { return m_colNum; }

    auto EvalRegister() const
    {
        using TEvalUnit = NSZeroMatrix::EvalUnit<ElementType, DeviceType>;
        using TEvalGroup = TrivalEvalGroup<TEvalUnit>;
        if (!m_evalBuf.IsEvaluated())
        {
            auto evalHandle = m_evalBuf.Handle();
            decltype(auto) outPtr = evalHandle.DataPtr();
            TEvalUnit unit(std::move(evalHandle), m_rowNum, m_colNum);
            EvalPlan<DeviceType>::template Register<TEvalGroup>(std::move(unit), outPtr, {});
        }
        return m_evalBuf.ConstHandle();
    }
    
private:
    size_t m_rowNum;
    size_t m_colNum;
    EvalBuffer<Matrix<ElementType, DeviceType>> m_evalBuf;
};

template <typename TElem, typename TDevice>
constexpr bool IsMatrix<ZeroMatrix<TElem, TDevice>> = true;
}