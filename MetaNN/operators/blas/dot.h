#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <MetaNN/operators/facilities/operator_frame.h>
#include <MetaNN/operators/facilities/tail_calculator.h>
#include <cassert>
#include <type_traits>

namespace MetaNN::OpTags
{
    struct Dot;
}

namespace MetaNN
{
namespace OperDot::NSCaseGen
{
    template <typename TInputHandle1, typename TInputHandle2, typename TOutputHandle>
    class EvalItem : public BaseEvalItem<DeviceTypeFromHandle<TOutputHandle>>
    {
        using BaseType = BaseEvalItem<DeviceTypeFromHandle<TOutputHandle>>;
    public:
        template <typename TAuxParams>
        EvalItem(TInputHandle1 operand1, TInputHandle2 operand2, 
                 TOutputHandle outputHandle, const TAuxParams&)
            : BaseType(std::type_index(typeid(EvalItem)),
                       {operand1.DataPtr(), operand2.DataPtr()},
                       outputHandle.DataPtr())
            , m_operand1(std::move(operand1))
            , m_operand2(std::move(operand2))
            , m_outputHandle(std::move(outputHandle))
        {}
        
        const TInputHandle1 m_operand1;
        const TInputHandle2 m_operand2;
        TOutputHandle m_outputHandle;
    };

    template <typename TInputHandle1, typename TInputHandle2, typename TOutputHandle>
    class EvalGroup : public TrivalEvalGroup<EvalItem<TInputHandle1, TInputHandle2, TOutputHandle>>
    {
        using EvalItemType = EvalItem<TInputHandle1, TInputHandle2, TOutputHandle>;
    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            const auto& in1 = evalItem.m_operand1.Data();
            const auto& in2 = evalItem.m_operand2.Data();

            auto aimShape = in1.Shape();
            aimShape.ColNum() = in2.Shape().ColNum();

            using ResType = typename TOutputHandle::DataType;
            using ElementType = typename ResType::ElementType;
            ResType out(aimShape);

            const size_t m = in1.Shape().RowNum();
            const size_t k = in1.Shape().ColNum();
            const size_t n = in2.Shape().ColNum();
            assert(k == in2.Shape().RowNum());

            const size_t loopCount = in1.Shape().Count() / m / k;
            assert(loopCount == in2.Shape().Count() / k / n);

            auto low_in1 = LowerAccess(in1);
            const ElementType* mem_in1 = low_in1.RawMemory();

            auto low_in2 = LowerAccess(in2);
            const ElementType* mem_in2 = low_in2.RawMemory();

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
            evalItem.m_outputHandle.SetData(std::move(out));
        }
    };
}

template <typename TOperand1, typename TOperand2>
constexpr bool IsValidOper<OpTags::Dot, TOperand1, TOperand2> =
    (IsMatrix<TOperand1> && IsMatrix<TOperand2>) ||
    (IsBatchMatrix<TOperand1> && IsBatchMatrix<TOperand2>) ||
    (IsMatrixSequence<TOperand1> && IsMatrixSequence<TOperand2>) ||
    (IsBatchMatrixSequence<TOperand1> && IsBatchMatrixSequence<TOperand2>);

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
    using type = OperCalAlgoChain<TailCalculator<OperDot::NSCaseGen::EvalItem, OperDot::NSCaseGen::EvalGroup>>;
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
