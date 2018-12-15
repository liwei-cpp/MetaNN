#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <MetaNN/operators/facilities/tags.h>
#include <MetaNN/operators/facilities/operator_frame.h>
#include <cassert>
#include <type_traits>

namespace MetaNN
{
namespace OperDot::NSCaseGen
{
template <typename TInputHandle1,
          typename TInputHandle2,
          typename TOutputHandle>
class EvalUnit : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
public:
    EvalUnit(TInputHandle1 operand1,
             TInputHandle2 operand2,
             TOutputHandle outputHandle)
        : m_operand1(std::move(operand1))
        , m_operand2(std::move(operand2))
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        const auto& in1 = m_operand1.Data();
        const auto& in2 = m_operand2.Data();

        auto aimShape = in1.Shape();
        aimShape.ColNum() = in2.Shape().ColNum();
        m_outputHandle.Allocate(aimShape);
        auto& out = m_outputHandle.MutableData();
        
        using ElementType = ElementTypePicker<decltype(out)>;
        
        const size_t m = in1.Shape().RowNum();
        const size_t k = in1.Shape().ColNum();
        const size_t n = in2.Shape().ColNum();
        assert(k == in2.Shape().RowNum());
        
        const size_t loopCount = in1.Shape().Count() / m / k;
        assert(loopCount == in2.Shape().Count() / k / n);
        
        auto low_in1 = LowerAccess(in1);
        ElementType* mem_in1 = low_in1.MutableRawMemory();
        
        auto low_in2 = LowerAccess(in2);
        ElementType* mem_in2 = low_in2.MutableRawMemory();

        auto low_out = LowerAccess(out);
        ElementType* mem_out = low_out.MutableRawMemory();
                
        static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
        
        for (size_t loop = 0; loop < loopCount; ++loop)
        {
            for (size_t i = 0; i < m; ++i)
            {
                for (size_t j = 0; j < n; ++j)
                {
                    mem_out[i * n + j] = 0;
                    for (size_t l = 0; l < k; ++l)
                    {
                        mem_out[i * n + j] += mem_in1[i * k + l] * mem_in2[l * n + j];
                    }
                }
            }
            mem_out += m * n;
            mem_in1 += m * k;
            mem_in2 += k * n;
        }
        m_outputHandle.SetEval();
    }
    
private:
    const TInputHandle1 m_operand1;
    const TInputHandle2 m_operand2;
    TOutputHandle m_outputHandle;
};
}

template <typename TOperand>
constexpr bool IsValidOper<OpTags::Dot, TOperand> =
    IsMatrix<TOperand> ||
    IsBatchMatrix<TOperand> ||
    IsMatrixSequence<TOperand> ||
    IsBatchMatrixSequence<TOperand>;

template <typename TCate>
class OperShapeInfo<OpTags::Dot, TCate>
{
public:
    template <typename TOperand1, typename TOperand2>
    OperShapeInfo(const OperAuxParams<OpTags::Dot, TCate>&,
                  const TOperand1& operand1,
                  const TOperand2& operand2)
        : m_shape(operand1.Shape())
    {
        m_shape.ColNum() = operand2.Shape().ColNum();
    }
    
    const auto& Shape() const
    {
        return m_shape;
    }
    
private:
    MetaNN::Shape<TCate> m_shape;
};

template <>
struct OperSeq_<OpTags::Dot>
{
    using type = OperSeqContainer<TailCalculator<OperDot::NSCaseGen::EvalUnit>>;
};

template <typename TP1, typename TP2,
          typename = std::enable_if_t<IsValidOper<OpTags::Dot, TP1, TP2>>>
auto Dot(TP1&& p_m1, TP2&& p_m2)
{
    if (p_m1.Shape().ColNum() != p_m2.Shape().RowNum())
    {
        throw std::runtime_error("Shape mismatch for Dot.");
    }
    auto checkShape = p_m1.Shape();
    checkShape.RowNum() = p_m2.Shape().RowNum();
    checkShape.ColNum() = p_m2.Shape().ColNum();
    if (checkShape != p_m2.Shape())
    {
        throw std::runtime_error("Shape mismatch for Dot.");
    }
    
    using ResType = Operator<OpTags::Dot, RemConstRef<TP1>, RemConstRef<TP2>>;
    return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
}
}
