#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <MetaNN/operators/activations/tags.h>
#include <MetaNN/operators/facilities/tail_calculator.h>
#include <cassert>
#include <cmath>
#include <type_traits>

namespace MetaNN
{
namespace OperSoftmax::NSCaseGen
{
template <typename TInputHandle, typename TOutputHandle>
class EvalUnit : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
    using ElementType = ElementTypeFromHandle<TOutputHandle>;
public:
    template <typename TAuxParams>
    EvalUnit(TInputHandle oriHandle, TOutputHandle outputHandle, const TAuxParams&)
        : m_inputHandle(std::move(oriHandle))
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        const auto& in = m_inputHandle.Data();
        m_outputHandle.Allocate(in.Shape());
        auto& out = m_outputHandle.MutableData();
        
        const size_t count = in.Shape().Count();
        assert(count == out.Shape().Count());
        const size_t matrixSize = in.Shape().RowNum() * in.Shape().ColNum();
        assert(count % matrixSize == 0);
        const size_t loopCount = count / matrixSize;
        
        auto low_in = LowerAccess(in);
        ElementType* mem_in = low_in.MutableRawMemory();

        auto low_out = LowerAccess(out);
        ElementType* mem_out = low_out.MutableRawMemory();
                
        static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
        
        for (size_t i = 0; i < loopCount; ++i)
        {
            EvalMatrix(mem_out, mem_in, matrixSize);
            mem_out += matrixSize;
            mem_in += matrixSize;
        }
        m_outputHandle.SetEval();
    }
    
private:
    void EvalMatrix(ElementType* out, ElementType* in, const size_t len)
    {
        ElementType maxElem = *std::max_element(in, in + len);
        ElementType sum{};

        for (size_t i = 0; i < len; ++i)
        {
            out[i] = exp(in[i] - maxElem);
            sum += out[i];
        }

        for (size_t i = 0; i < len; ++i)
        {
            out[i] /= sum;
        }
    }
private:
    const TInputHandle m_inputHandle;
    TOutputHandle m_outputHandle;
};
}

template <typename TOperand>
constexpr bool IsValidOper<OpTags::Softmax, TOperand> =
    IsMatrix<TOperand> ||
    IsBatchMatrix<TOperand>;

template <>
struct OperSeq_<OpTags::Softmax>
{
    using type = OperSeqContainer<TailCalculator<OperSoftmax::NSCaseGen::EvalUnit>>;
};

template <typename TP,
          typename = std::enable_if_t<IsValidOper<OpTags::Softmax, TP>>>
auto Softmax(TP&& p_m)
{
    using rawM = RemConstRef<TP>;
    using ResType = Operator<OpTags::Softmax, rawM>;
    return ResType(std::forward<TP>(p_m));
}
}