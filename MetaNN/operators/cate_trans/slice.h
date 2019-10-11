#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <MetaNN/operators/facilities/tail_calculator.h>
#include <cassert>
#include <type_traits>

namespace MetaNN::OpTags
{
    struct Slice;
}

namespace MetaNN
{
namespace OperSlice::NSCaseGen
{
    template <typename TInputHandle, typename TOutputHandle>
    class EvalUnit : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
    {
    public:
        template <typename TAuxParams>
        EvalUnit(TInputHandle oriHandle, TOutputHandle outputHandle, const TAuxParams& p_params)
            : m_inputHandle(std::move(oriHandle))
            , m_id(p_params.m_elemID)
            , m_outputHandle(std::move(outputHandle))
        {}
    
        void Eval() override final
        {
            using InputDatType = typename TInputHandle::DataType;
            const InputDatType& in = m_inputHandle.Data();
            m_outputHandle.SetData(in[m_id]);
        }
    
    private:
        const TInputHandle m_inputHandle;
        const size_t m_id;
        TOutputHandle m_outputHandle;
    };
}

    template <typename TOperand>
    constexpr bool IsValidOper<OpTags::Slice, TOperand> =
        IsBatchScalar<TOperand> || IsBatchMatrix<TOperand> || IsBatchThreeDArray<TOperand> ||
        IsScalarSequence<TOperand> || IsMatrixSequence<TOperand> || IsThreeDArraySequence<TOperand> ||
        IsBatchScalarSequence<TOperand> || IsBatchMatrixSequence<TOperand> || IsBatchThreeDArraySequence<TOperand>;
        
    template <typename TPrimaryCate>
    struct OperCategory_<OpTags::Slice, CategoryTags::Batch<TPrimaryCate>>
    {
        using type = TPrimaryCate;
    };
    
    template <typename TPrimaryCate>
    struct OperCategory_<OpTags::Slice, CategoryTags::Sequence<TPrimaryCate>>
    {
        using type = TPrimaryCate;
    };
    
    template <typename TPrimaryCate>
    struct OperCategory_<OpTags::Slice, CategoryTags::BatchSequence<TPrimaryCate>>
    {
        using type = CategoryTags::Sequence<TPrimaryCate>;
    };
    
    template <typename TCate>
    struct OperAuxParams<OpTags::Slice, TCate>
    {
        OperAuxParams(size_t p_elemID)
            : m_elemID(p_elemID) {}
        
        const size_t m_elemID;
        
        bool operator == (const OperAuxParams& val) const
        {
            return (m_elemID == val.m_elemID);
        }
    };
    
    template <typename TCate>
    class OperShapeInfo<OpTags::Slice, TCate>
    {
    public:
        template <typename TOperand>
        OperShapeInfo(const OperAuxParams<OpTags::Slice, TCate>&, const TOperand& operand)
            : m_shape(operand.Shape().Cardinal())
        { }
    
        const auto& Shape() const
        {
            return m_shape;
        }
    
    private:
        MetaNN::Shape<TCate> m_shape;
    };
    
    template <typename TCate>
    class OperShapeInfo<OpTags::Slice, CategoryTags::Sequence<TCate>>
    {
    public:
        template <typename TOperand>
        OperShapeInfo(const OperAuxParams<OpTags::Slice, CategoryTags::Sequence<TCate>>& param, const TOperand& operand)
            : m_shape(operand.Shape().SeqLenContainer()[param.m_elemID], operand.Shape().Cardinal())
        { }
    
        const auto& Shape() const
        {
            return m_shape;
        }
    
    private:
        MetaNN::Shape<CategoryTags::Sequence<TCate>> m_shape;
    };
    
    template <>
    struct OperSeq_<OpTags::Slice>
    {
        using type = OperSeqContainer<TailCalculator<OperSlice::NSCaseGen::EvalUnit>>;
    };

    template <typename TOpTag, typename...TOperands>
    auto Operator<TOpTag, TOperands...>::operator[](size_t p_index) const -> Operator<OpTags::Slice, Operator>
    {
        if constexpr (IsValidOper<OpTags::Slice, Operator>)
        {
            using ResType = Operator<OpTags::Slice, Operator>;
            return ResType(OperAuxParams<OpTags::Slice, typename ResType::CategoryTag>(p_index), (const Operator&)*this);
        }
        else
        {
            static_assert(DependencyFalse<Operator>, "Operator not support slice.");
        }
    }
}