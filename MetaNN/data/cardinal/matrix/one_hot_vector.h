#pragma once
#include <cstring>
#include <MetaNN/data/cardinal/matrix/matrix.h>
#include <MetaNN/evaluate/eval_buffer.h>
#include <MetaNN/evaluate/eval_handle.h>
#include <MetaNN/evaluate/eval_plan.h>

namespace MetaNN
{
namespace NSOneHotVector
{
    template <typename TElem, typename TDevice>
    class EvalItem : public BaseEvalItem<TDevice>
    {
    public:
        EvalItem(EvalHandle<Matrix<TElem, TDevice>> resBuf,
                 size_t colNum, size_t val)
            : BaseEvalItem<TDevice>(std::type_index(typeid(EvalItem)),
                                    {}, resBuf.DataPtr())
            , m_resHandle(std::move(resBuf))
            , m_colNum(colNum)
            , m_val(val)
        {}
        
    public:
        EvalHandle<Matrix<TElem, TDevice>> m_resHandle;
        size_t m_colNum;
        size_t m_val;
    };
    
    template <typename TElem, typename TDevice>
    class EvalGroup : public TrivalEvalGroup<EvalItem<TElem, TDevice>>
    {
        using EvalItemType = EvalItem<TElem, TDevice>;
    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            Matrix<TElem, TDevice> out(1, evalItem.m_colNum);
        
            static_assert(std::is_same_v<TDevice, DeviceTags::CPU>,
                          "Currently only CPU is supported.");
            auto lowLayer = LowerAccess(out);
            auto mem = lowLayer.MutableRawMemory();
            memset(mem, 0, sizeof(TElem) * evalItem.m_colNum);
            mem[evalItem.m_val] = 1;
            evalItem.m_resHandle.SetData(std::move(out));
        }
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
        using TEvalItem = NSOneHotVector::EvalItem<ElementType, DeviceType>;
        using TEvalGroup = NSOneHotVector::EvalGroup<ElementType, DeviceType>;
        using TItemDispatcher = TrivalEvalItemDispatcher<TEvalGroup>;
        
        if (!m_evalBuf.IsEvaluated())
        {
            auto evalHandle = m_evalBuf.Handle();
            if (!EvalPlan<DeviceType>::Inst().IsAlreayRegisted(evalHandle.DataPtr()))
            {
                EvalPlan<DeviceType>::Inst().template Register<TItemDispatcher>(
                    std::make_unique<TEvalItem>(std::move(evalHandle), m_shape.ColNum(), m_hotPos));
            }
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
