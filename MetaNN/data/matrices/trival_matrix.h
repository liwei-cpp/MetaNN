#pragma once

#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/data/matrices/matrices.h>
#include <MetaNN/evaluate/facilities/eval_buffer.h>
#include <MetaNN/evaluate/facilities/eval_group.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <cassert>
#include <memory>

namespace MetaNN
{
namespace NSTrivalMatrix
{
template <typename TElem, typename TDevice>
class EvalUnit;

template <typename TElem>
class EvalUnit<TElem, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    template <typename TScaleElemType>
    EvalUnit(EvalHandle<Matrix<TElem, DeviceTags::CPU>> resBuf,
             size_t rowNum, size_t colNum,
             const Scalar<TScaleElemType, DeviceTags::CPU>& val)
        : m_resHandle(std::move(resBuf))
        , m_rowNum(rowNum)
        , m_colNum(colNum)
        , m_val(val.Value()) {}

    void Eval() override
    {
        m_resHandle.Allocate(m_rowNum, m_colNum);
        auto& mutableData = m_resHandle.MutableData();
        auto lowLayer = LowerAccess(mutableData);
        const size_t rowLen = lowLayer.RowLen();
        auto mem = lowLayer.MutableRawMemory();
        for (size_t i = 0; i < m_rowNum; ++i)
        {
            for (size_t j = 0; j < m_colNum; ++j)
            {
                mem[j] = m_val;
            }
            mem += rowLen;
        }
        m_resHandle.SetEval();
    }

private:
    EvalHandle<Matrix<TElem, DeviceTags::CPU>> m_resHandle;
    size_t m_rowNum;
    size_t m_colNum;
    TElem  m_val;
};
}

template<typename TElem, typename TDevice, typename TScalar>
class TrivalMatrix
{
public:
    using ElementType = TElem;
    using DeviceType = TDevice;
    
public:
    TrivalMatrix(size_t p_rowNum, size_t p_colNum,
                 TScalar p_val)
        : m_val(p_val)
        , m_rowNum(p_rowNum)
        , m_colNum(p_colNum) {}

    bool operator== (const TrivalMatrix& val) const
    {
        return (m_val == val.m_val) &&
               (m_rowNum == val.m_rowNum) &&
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

    size_t RowNum() const
    {
        return m_rowNum;
    }

    size_t ColNum() const
    {
        return m_colNum;
    }

    auto EvalRegister() const
    {
        using TEvalUnit = NSTrivalMatrix::EvalUnit<ElementType, DeviceType>;
        using TEvalGroup = TrivalEvalGroup<TEvalUnit>;
        if (!m_evalBuf.IsEvaluated())
        {
            auto evalHandle = m_evalBuf.Handle();
            const void* outputPtr = evalHandle.DataPtr();
            TEvalUnit unit(std::move(evalHandle), m_rowNum, m_colNum, m_val);
            EvalPlan<DeviceType>::template Register<TEvalGroup>(std::move(unit), outputPtr, {});
        }
        return m_evalBuf.ConstHandle();
    }

    auto ElementValue() const
    {
        return m_val;
    }

private:
    TScalar m_val;
    size_t m_rowNum;
    size_t m_colNum;
    EvalBuffer<Matrix<ElementType, DeviceType>> m_evalBuf;
};

template <typename TElem, typename TDevice, typename TScalar>
constexpr bool IsMatrix<TrivalMatrix<TElem, TDevice, TScalar>> = true;

template<typename TElem, typename TDevice, typename TVal>
auto MakeTrivalMatrix(size_t rowNum, size_t colNum, TVal&& m_val)
{
    using RawVal = RemConstRef<TVal>;
    
    if constexpr (IsScalar<RawVal>)
    {
        static_assert(std::is_same<typename RawVal::DeviceType, TDevice>::value ||
                      std::is_same<typename RawVal::DeviceType, DeviceTags::CPU>::value);
        return TrivalMatrix<TElem, TDevice, RawVal>(rowNum, colNum, std::forward<TVal>(m_val));
    }
    else
    {
        TElem tmpElem = static_cast<TElem>(m_val);
        Scalar<TElem, DeviceTags::CPU> scalar(std::move(tmpElem));
        return TrivalMatrix<TElem, TDevice, Scalar<TElem, DeviceTags::CPU>>(rowNum, colNum, std::move(scalar));
    }
}
}
