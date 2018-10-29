#pragma once

#include <MetaNN/data/cardinal/matrix/matrix_base.h>
#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/evaluate/facilities/eval_buffer.h>
#include <MetaNN/evaluate/facilities/eval_group.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>

#include <cassert>
#include <memory>

namespace MetaNN
{
namespace NSOneHotVector
{
template <typename TElem, typename TDevice>
class EvalUnit;

template <typename TElement>
class EvalUnit<TElement, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    EvalUnit(EvalHandle<Matrix<TElement, DeviceTags::CPU>> resBuf,
             size_t colNum, size_t val)
        : m_resHandle(std::move(resBuf))
        , m_colNum(colNum)
        , m_val(val)
    {
        assert(m_val < m_colNum);
    }

    void Eval() override
    {
        auto& mutableData = m_resHandle.MutableData();
        m_resHandle.Allocate(1, m_colNum);
        auto lowLayer = LowerAccess(mutableData);
        auto mem = lowLayer.MutableRawMemory();
        memset(mem, 0, sizeof(TElement) * m_colNum);
        mem[m_val] = 1;
        m_resHandle.SetEval();
    }

private:
    EvalHandle<Matrix<TElement, DeviceTags::CPU>> m_resHandle;
    size_t m_colNum;
    size_t m_val;
};
}

template <typename TElem, typename TDevice>
class OneHotVector : public Shape<CategoryTags::Matrix>
{
public:
    using ElementType = TElem;
    using DeviceType = TDevice;

public:
    OneHotVector(size_t p_colNum, size_t p_hotPos)
        : Shape<CategoryTags::Matrix>(1, p_colNum)
        , m_hotPos(p_hotPos)
    {
        assert(p_hotPos < ColNum());
    }

    bool operator== (const OneHotVector& val) const
    {
        return (GetShape() == val.GetShape()) &&
               (m_hotPos == val.m_hotPos);
    }

    template <typename TOtherType,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<TOtherType>, OneHotVector>>>
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
        using TEvalUnit = NSOneHotVector::EvalUnit<ElementType, DeviceType>;
        using TEvalGroup = TrivalEvalGroup<TEvalUnit>;
        if (!m_evalBuf.IsEvaluated())
        {
            auto evalHandle = m_evalBuf.Handle();
            decltype(auto) outputPtr = evalHandle.DataPtr();
            TEvalUnit unit(std::move(evalHandle), ColNum(), m_hotPos);
            EvalPlan<DeviceType>::template Register<TEvalGroup>(std::move(unit), outputPtr, {});
        }
        return m_evalBuf.ConstHandle();
    }

    auto HotPos() const noexcept
    {
        return m_hotPos;
    }

private:
    size_t m_hotPos;
    EvalBuffer<Matrix<ElementType, DeviceType>> m_evalBuf;
};

template <typename TElem, typename TDevice>
struct DataCategory_<OneHotVector<TElem, TDevice>>
{
    using type = CategoryTags::Matrix;
};
}