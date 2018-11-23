#pragma once
#include <MetaNN/data/cardinal/matrix/matrix.h>
#include <MetaNN/evaluate/facilities/eval_buffer.h>
#include <MetaNN/evaluate/facilities/eval_group.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>

namespace MetaNN
{
namespace NSOneHotVector
{
template <typename TElem, typename TDevice>
class EvalUnit : public BaseEvalUnit<TDevice>
{
public:
    EvalUnit(EvalHandle<Matrix<TElem, TDevice>> resBuf,
             size_t colNum, size_t val)
        : m_resHandle(std::move(resBuf))
        , m_colNum(colNum)
        , m_val(val)
    {
        assert(m_val < m_colNum);
    }
    
    void Eval() override final
    {
        auto& mutableData = m_resHandle.MutableData();
        m_resHandle.Allocate(MetaNN::Shape<CategoryTags::Matrix>{1, m_colNum});
        
        static_assert(std::is_same_v<TDevice, DeviceTags::CPU>,
                      "Currently only CPU is supported.");
        auto lowLayer = LowerAccess(mutableData);
        auto mem = lowLayer.MutableRawMemory();
        memset(mem, 0, sizeof(TElem) * m_colNum);
        mem[m_val] = 1;
        m_resHandle.SetEval();
    }

private:
    EvalHandle<Matrix<TElem, TDevice>> m_resHandle;
    size_t m_colNum;
    size_t m_val;
};
}

template <typename TElem, typename TDevice>
class OneHotVector
{
public:
    using CategoryTag = CategoryTags::Matrix;
    using ElementType = TElem;
    using DeviceType = TDevice;

public:
    explicit OneHotVector(size_t colNum, size_t p_hotPos)
        : OneHotVector(MetaNN::Shape<CategoryTag>{1, colNum}, p_hotPos)
    {}
    
    explicit OneHotVector(MetaNN::Shape<CategoryTag> p_shape, size_t p_hotPos)
        : m_shape(std::move(p_shape))
        , m_hotPos(p_hotPos)
    {
        if (m_shape.RowNum() != 1)
        {
            throw std::runtime_error("One hot vector must have 1 row.");
        }
        if (p_hotPos >= m_shape.ColNum())
        {
            throw std::runtime_error("One hot vector hot position setting error.");
        }
    }
    
    const auto& Shape() const noexcept
    {
        return m_shape;
    }
    
    bool operator== (const OneHotVector& val) const
    {
        return (m_shape == val.m_shape) &&
               (m_hotPos == val.m_hotPos);
    }
    
    auto EvalRegister() const
    {
        using TEvalUnit = NSOneHotVector::EvalUnit<ElementType, DeviceType>;
        using TEvalGroup = TrivalEvalGroup<TEvalUnit>;
        if (!m_evalBuf.IsEvaluated())
        {
            auto evalHandle = m_evalBuf.Handle();
            decltype(auto) outputPtr = evalHandle.DataPtr();
            TEvalUnit unit(std::move(evalHandle), m_shape.ColNum(), m_hotPos);
            EvalPlan<DeviceType>::template Register<TEvalGroup>(std::move(unit), outputPtr, {});
        }
        return m_evalBuf.ConstHandle();
    }

    auto HotPos() const noexcept
    {
        return m_hotPos;
    }

private:
    MetaNN::Shape<CategoryTag> m_shape;
    size_t m_hotPos;
    EvalBuffer<Matrix<TElem, TDevice>> m_evalBuf;
};
}
