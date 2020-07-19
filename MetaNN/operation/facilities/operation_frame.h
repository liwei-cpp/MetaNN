#pragma once

#include <cassert>
#include <type_traits>
#include <MetaNN/evaluate/eval_buffer.h>
#include <MetaNN/facilities/cont_metafuns/sequential.h>
#include <MetaNN/operation/facilities/organizer.h>
#include <MetaNN/policies/policy_container.h>

namespace MetaNN::OpTags
{
    struct Slice;
}

namespace MetaNN
{
    template <typename TOperand>
    constexpr bool IsValidOper<OpTags::Slice, TOperand> = (DataCategory<TOperand>::DimNum > 0);

    template <typename TOpTag, typename TOperands, typename TPolicies = PolicyContainer<>>
    class Operation;
    
    template <typename TOpTag, typename TPolicies, typename... TOperands>
    class Operation<TOpTag, OperandContainer<TOperands...>, TPolicies>
    {
        static_assert(sizeof...(TOperands) > 0, "Operation not support zero operands.");
        static_assert((std::is_same_v<RemConstRef<TOperands>, TOperands> && ...),
                      "TOperands is not an available types");
    public:
        using Policies = TPolicies;
        using CategoryTag = OperCateCal<TOpTag, TPolicies, TOperands...>;
        using ElementType = typename OperElementType_<TOpTag, TOperands...>::type;
        using DeviceType = typename OperDeviceType_<TOpTag, TOperands...>::type;

        template <size_t Id>
        using OperandType = Sequential::At<OperandContainer<TOperands...>, Id>;

    public:
        explicit Operation(TOperands... p_operands)
            : Operation(OperAuxParams<TOpTag, ElementType, CategoryTag>{},
                        std::move(p_operands)...)
        {}
    
        explicit Operation(OperAuxParams<TOpTag, ElementType, CategoryTag> auxParams,
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

        bool operator== (const Operation& val) const
        {
            // Note: shape is deduced from m_auxParams and m_operands, so if they are same, then shape is same.
            return (m_auxParams == val.m_auxParams) &&
                   (m_operands == val.m_operands);
        }

        auto operator[](size_t index) const
        {
            if constexpr (IsValidOper<OpTags::Slice, Operation>)
            {
                using ResType = Operation<OpTags::Slice, OperandContainer<Operation>>;
                return ResType(OperAuxParams<OpTags::Slice,
                                             typename ResType::ElementType,
                                             typename ResType::CategoryTag>(index), (const Operation&)*this);
            }
            else
            {
                static_assert(DependencyFalse<Operation>, "Slice is not supported.");
            }
        }

        auto EvalRegister() const
        {
            if (!m_evalBuf.IsEvaluated())
            {
                auto evalHandle = m_evalBuf.Handle();
                if (!EvalPlan::Inst().IsAlreadyRegisted(evalHandle.DataPtr()))
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
        OperAuxParams<TOpTag, ElementType, CategoryTag> m_auxParams;
        OperShapeInfo<TOpTag, CategoryTag, TPolicies> m_shapeInfo;
        std::tuple<TOperands...> m_operands;

        using TPrincipal = PrincipalDataType<CategoryTag, ElementType, DeviceType>;
        EvalBuffer<TPrincipal> m_evalBuf;
    };
}