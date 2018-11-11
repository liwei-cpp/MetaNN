#pragma once

#include <MetaNN/data/cardinal/matrix/matrix.h>
#include <MetaNN/evaluate/facilities/eval_buffer.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>
#include <MetaNN/evaluate/facilities/eval_group.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>

namespace MetaNN
{
namespace NSTrivalMatrix
{
template <typename TElem, typename TDevice, typename TScalar>
class EvalUnit : public BaseEvalUnit<TDevice>
{
public:
    EvalUnit(EvalHandle<Matrix<TElem, TDevice>> resBuf,
             Shape<CategoryTags::Matrix> p_shape,
             TScalar p_scalar)
        : m_resHandle(std::move(resBuf))
        , m_shape(std::move(p_shape))
        , m_scalar(std::move(p_scalar))
    {}

    void Eval() override final
    {
        static_assert(std::is_same_v<TDevice, DeviceTags::CPU> &&
                      std::is_same_v<typename TScalar::DeviceType, DeviceTags::CPU>,
                      "Currently only CPU is supported.");

        m_resHandle.Allocate(m_shape);
        auto& mutableData = m_resHandle.MutableData();
        auto lowLayer = LowerAccess(mutableData);
        auto mem = lowLayer.MutableRawMemory();
        
        const size_t elemCount = m_shape.Count();
        const TElem val = static_cast<TElem>(m_scalar.Value());
        for (size_t i = 0; i < elemCount; ++i)
        {
            mem[i] = val;
        }
        m_resHandle.SetEval();
    }

private:
    EvalHandle<Matrix<TElem, TDevice>> m_resHandle;
    Shape<CategoryTags::Matrix> m_shape;
    TScalar  m_scalar;
};
}

template<typename TElem, typename TDevice, typename TScalar>
class TrivalMatrix
{
    static_assert(std::is_same<RemConstRef<TElem>, TElem>::value,
                  "TElem is not a valid type");
    static_assert(std::is_same<RemConstRef<TScalar>, TScalar>::value,
                  "TScalar is not a valid type");
public:
    using CategoryTag = CategoryTags::Matrix;
    using ElementType = TElem;
    using DeviceType = TDevice;

public:
    explicit TrivalMatrix(TScalar p_scalar, size_t rowNum, size_t colNum)
        : m_shape(rowNum, colNum)
        , m_scalar(std::move(p_scalar))
    {}
    
    explicit TrivalMatrix(TScalar p_scalar, MetaNN::Shape<CategoryTag> p_shape)
        : m_shape(std::move(p_shape))
        , m_scalar(std::move(p_scalar))
    {}
    
    const auto& Shape() const noexcept
    {
        return m_shape;
    }
    
    bool operator== (const TrivalMatrix& val) const
    {
        return (m_shape == val.m_shape) &&
               (m_scalar == val.m_scalar);
    }

    auto EvalRegister() const
    {
        using TEvalUnit = NSTrivalMatrix::EvalUnit<ElementType, DeviceType, TScalar>;
        using TEvalGroup = TrivalEvalGroup<TEvalUnit>;
        if (!m_evalBuf.IsEvaluated())
        {
            auto evalHandle = m_evalBuf.Handle();
            const void* outputPtr = evalHandle.DataPtr();
            TEvalUnit unit(std::move(evalHandle), m_shape, m_scalar);
            EvalPlan<DeviceType>::template Register<TEvalGroup>(std::move(unit), outputPtr, {});
        }
        return m_evalBuf.ConstHandle();
    }

    const auto& ElementValue() const
    {
        return m_scalar;
    }

private:
    MetaNN::Shape<CategoryTag> m_shape;
    TScalar  m_scalar;
    EvalBuffer<Matrix<ElementType, DeviceType>> m_evalBuf;
};

template<typename TElem, typename TDevice, typename TVal, typename... TShapeParams>
auto MakeTrivalMatrix(TVal&& m_val, TShapeParams&&... shapeParams)
{
    using RawVal = RemConstRef<TVal>;
        
    MetaNN::Shape<CategoryTags::Matrix> shape(std::forward<TShapeParams>(shapeParams)...);
    if constexpr (IsScalar<RawVal>)
    {
        static_assert(std::is_same<typename RawVal::DeviceType, TDevice>::value ||
                      std::is_same<typename RawVal::DeviceType, DeviceTags::CPU>::value);
        return TrivalMatrix<TElem, TDevice, RawVal>(std::forward<TVal>(m_val), std::move(shape));
    }
    else
    {
        TElem tmpElem = static_cast<TElem>(m_val);
        Scalar<TElem, DeviceTags::CPU> scalar(std::move(tmpElem));
        return TrivalMatrix<TElem, TDevice, Scalar<TElem, DeviceTags::CPU>>(std::move(scalar), std::move(shape));
    }
}
}
