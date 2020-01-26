#pragma once

#include <cassert>
#include <type_traits>
#include <MetaNN/evaluate/eval_buffer.h>
#include <MetaNN/facilities/cont_metafuns/sequential.h>
#include <MetaNN/operators/facilities/organizer.h>

namespace MetaNN::OpTags
{
    struct Slice;
}

namespace MetaNN
{
template <typename TOpTag, typename...TOperands>
class Operator
{
    static_assert(sizeof...(TOperands) > 0, "Operator not support zero operands.");
    static_assert((std::is_same_v<RemConstRef<TOperands>, TOperands> && ...),
                  "TOperands is not available types");
public:
    using CategoryTag = OperCateCal<TOpTag, TOperands...>;
    using ElementType = typename OperElementType_<TOpTag, TOperands...>::type;
    using DeviceType = typename OperDeviceType_<TOpTag, TOperands...>::type;
    
public:
    explicit Operator(TOperands... p_operands)
        : Operator(OperAuxParams<TOpTag, CategoryTag>{},
                   std::move(p_operands)...)
    {}
    
    explicit Operator(OperAuxParams<TOpTag, CategoryTag> auxParams,
                      TOperands... p_operands)
        : m_auxParams(std::move(auxParams))
        , m_shapeInfo(m_auxParams, p_operands...)
        , m_operands({std::move(p_operands)...})
    {}
    
    template <size_t id>
    const auto& Operand() const
    {
        return std::get<id>(m_operands);
    }
    
    const auto& OperandTuple() const noexcept
    {
        return m_operands;
    }
    
    const auto& AuxParams() const
    {
        return m_auxParams;
    }
    
    const auto& Shape() const 
    {
        return m_shapeInfo.Shape();
    }
    
    bool operator== (const Operator& val) const
    {
        // Note: shape is deduced from m_auxParams and m_operands, so if they are same, then shape is same.
        return (m_auxParams == val.m_auxParams) &&
               (m_operands == val.m_operands);
    }
    
    Operator<OpTags::Slice, Operator> operator[](size_t index) const;

    auto EvalRegister() const
    {
        if (!m_evalBuf.IsEvaluated())
        {
            auto evalHandle = m_evalBuf.Handle();
            if (!EvalPlan<DeviceType>::Inst().IsAlreayRegisted(evalHandle.DataPtr()))
            {
                using TOperSeqCont = typename OperSeq_<TOpTag>::type;
            
                using THead = Sequential::Head<TOperSeqCont>;
                using TTail = Sequential::Tail<TOperSeqCont>;
                THead::template EvalRegister<TTail>(m_evalBuf, *this);
            }
        }
        return m_evalBuf.ConstHandle();
    }
    
private:
    OperAuxParams<TOpTag, CategoryTag> m_auxParams;
    OperShapeInfo<TOpTag, CategoryTag> m_shapeInfo;
    std::tuple<TOperands...> m_operands;
    
    using TPrincipal = PrincipalDataType<CategoryTag, ElementType, DeviceType>;
    EvalBuffer<TPrincipal> m_evalBuf;
};
}