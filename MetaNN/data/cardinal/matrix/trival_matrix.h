#pragma once

#include <MetaNN/data/cardinal/matrix/matrix.h>
#include <MetaNN/evaluate/eval_buffer.h>
#include <MetaNN/evaluate/eval_handle.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <typeindex>

namespace MetaNN
{
namespace NSTrivalMatrix
{
    template <typename TElem, typename TDevice, typename TScalar>
    class EvalItem : public BaseEvalItem<TDevice>
    {
    public:
        EvalItem(EvalHandle<Matrix<TElem, TDevice>> resBuf,
                 Shape<CategoryTags::Matrix> p_shape,
                 TScalar p_scalar)
            : BaseEvalItem<TDevice>(std::type_index(typeid(EvalItem)),
                                    {}, resBuf.DataPtr())
            , m_resHandle(std::move(resBuf))
            , m_shape(std::move(p_shape))
            , m_scalar(std::move(p_scalar))
        { }
        
        EvalHandle<Matrix<TElem, TDevice>> m_resHandle;
        Shape<CategoryTags::Matrix> m_shape;
        TScalar  m_scalar;
    };
    
    template <typename TElem, typename TDevice, typename TScalar>
    class EvalGroup : public TrivalEvalGroup<EvalItem<TElem, TDevice, TScalar>>
    {
        using EvalItemType = EvalItem<TElem, TDevice, TScalar>;
    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            static_assert(std::is_same_v<TDevice, DeviceTags::CPU> &&
                          std::is_same_v<typename TScalar::DeviceType, DeviceTags::CPU>,
                          "Currently only CPU is supported.");

            Matrix<TElem, TDevice> out(evalItem.m_shape);
            auto lowLayer = LowerAccess(out);
            auto mem = lowLayer.MutableRawMemory();
        
            const size_t elemCount = evalItem.m_shape.Count();
            const TElem val = static_cast<TElem>(evalItem.m_scalar.Value());
            for (size_t i = 0; i < elemCount; ++i)
            {
                mem[i] = val;
            }
            evalItem.m_resHandle.SetData(std::move(out));
        }
    };
}

template<typename TScalar>
class TrivalMatrix
{
public:
    using CategoryTag = CategoryTags::Matrix;
    using ElementType = typename TScalar::ElementType;
    using DeviceType = typename TScalar::DeviceType;

public:
    template <typename...TParams>
    explicit TrivalMatrix(TScalar p_scalar, TParams&&... params)
        : m_shape(std::forward<TParams>(params)...)
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
        using TEvalItem = NSTrivalMatrix::EvalItem<ElementType, DeviceType, TScalar>;
        using TEvalGroup = NSTrivalMatrix::EvalGroup<ElementType, DeviceType, TScalar>;
        using TItemDispatcher = TrivalEvalItemDispatcher<TEvalGroup>;
        
        if (!m_evalBuf.IsEvaluated())
        {
            auto evalHandle = m_evalBuf.Handle();
            if (!EvalPlan<DeviceType>::Inst().IsAlreayRegisted(evalHandle.DataPtr()))
            {
                EvalPlan<DeviceType>::Inst().template Register<TItemDispatcher>(
                    std::make_unique<TEvalItem>(std::move(evalHandle), m_shape, m_scalar));
            }
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
}
