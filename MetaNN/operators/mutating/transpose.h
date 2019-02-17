#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <MetaNN/operators/facilities/operator_frame.h>
#include <cassert>
#include <type_traits>

namespace MetaNN::OpTags
{
    struct Transpose;
}

namespace MetaNN
{
namespace OperTranspose::NSCaseGen
{
template <typename TInputHandle, typename TOutputHandle>
class EvalUnit : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
public:
    template <typename TAuxParams>
    EvalUnit(TInputHandle oriHandle, TOutputHandle outputHandle, const TAuxParams&)
        : m_inputHandle(std::move(oriHandle))
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        const auto& in = m_inputHandle.Data();

        auto aimShape = in.Shape();
        std::swap(aimShape.RowNum(), aimShape.ColNum());
        m_outputHandle.Allocate(aimShape);
        auto& out = m_outputHandle.MutableData();
        
        using ElementType = ElementTypePicker<decltype(out)>;
        
        const size_t count = in.Shape().Count();
        assert(count == out.Shape().Count());
        
        auto low_in = LowerAccess(in);
        ElementType* mem_in = low_in.MutableRawMemory();

        auto low_out = LowerAccess(out);
        ElementType* mem_out = low_out.MutableRawMemory();
                
        static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
        
        const size_t oriColSize = in.Shape().ColNum();
        const size_t oriRowSize = in.Shape().RowNum();
        const size_t matrixSize = oriRowSize * oriColSize;

        assert(count % matrixSize == 0);
        const size_t loopCount = count / matrixSize;

        for (size_t loop = 0; loop < loopCount; ++loop)
        {
            for (size_t i = 0; i < oriRowSize; ++i)
            {
                for (size_t j = 0; j < oriColSize; ++j)
                {
                    mem_out[j * oriRowSize + i] = mem_in[i * oriColSize + j];
                }
            }
            mem_out += matrixSize;
            mem_in += matrixSize;
        }
        m_outputHandle.SetEval();
    }
    
private:
    const TInputHandle m_inputHandle;
    TOutputHandle m_outputHandle;
};
}

template <typename TOperand>
constexpr bool IsValidOper<OpTags::Transpose, TOperand> =
    IsMatrix<TOperand> ||
    IsBatchMatrix<TOperand> ||
    IsMatrixSequence<TOperand> ||
    IsBatchMatrixSequence<TOperand>;

template <typename TCate>
class OperShapeInfo<OpTags::Transpose, TCate>
{
public:
    template <typename TOperand>
    OperShapeInfo(const OperAuxParams<OpTags::Transpose, TCate>&, const TOperand& operand)
        : m_shape(operand.Shape())
    {
        std::swap(m_shape.RowNum(), m_shape.ColNum());
    }
    
    const auto& Shape() const
    {
        return m_shape;
    }
    
private:
    MetaNN::Shape<TCate> m_shape;
};

template <>
struct OperSeq_<OpTags::Transpose>
{
    using type = OperSeqContainer<TailCalculator<OperTranspose::NSCaseGen::EvalUnit>>;
};

template <typename TP,
          typename = std::enable_if_t<IsValidOper<OpTags::Transpose, TP>>>
auto Transpose(TP&& p_m)
{
    using rawM = RemConstRef<TP>;
    using ResType = Operator<OpTags::Transpose, rawM>;
    return ResType(std::forward<TP>(p_m));
}
}
