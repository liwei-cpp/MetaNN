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
        auto mem = lowLayer.MutableRawMemory();
        for (size_t i = 0; i < m_rowNum; ++i)
        {
            for (size_t j = 0; j < m_colNum; ++j)
            {
                mem[j] = m_val;
            }
            mem += m_colNum;
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
class TrivalMatrix : public Shape_<CategoryTags::Matrix>
{
public:
    using ElementType = TElem;
    using DeviceType = TDevice;
    
public:
    TrivalMatrix(size_t p_rowNum, size_t p_colNum,
                 TScalar p_val)
        : Shape_<CategoryTags::Matrix>(p_rowNum, p_colNum)
        , m_val(p_val)
    {}

    bool operator== (const TrivalMatrix& val) const
    {
        return (Shape() == val.Shape()) &&
               (m_val == val.m_val);
    }

    template <typename TOtherType,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<TOtherType>, TrivalMatrix>>>
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
        using TEvalUnit = NSTrivalMatrix::EvalUnit<ElementType, DeviceType>;
        using TEvalGroup = TrivalEvalGroup<TEvalUnit>;
        if (!m_evalBuf.IsEvaluated())
        {
            auto evalHandle = m_evalBuf.Handle();
            const void* outputPtr = evalHandle.DataPtr();
            TEvalUnit unit(std::move(evalHandle), RowNum(), ColNum(), m_val);
            EvalPlan<DeviceType>::template Register<TEvalGroup>(std::move(unit), outputPtr, {});
        }
        return m_evalBuf.ConstHandle();
    }

    const auto ElementValue() const
    {
        return m_val;
    }

private:
    TScalar m_val;
    EvalBuffer<Matrix<ElementType, DeviceType>> m_evalBuf;
};

template <typename TElem, typename TDevice, typename TScalar>
struct DataCategory_<TrivalMatrix<TElem, TDevice, TScalar>>
{
    using type = CategoryTags::Matrix;
};

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
