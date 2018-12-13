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
namespace OperSigmoid::NSCaseGen
{
template <typename TInputHandle, typename TOutputHandle>
class EvalUnit : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
public:
    EvalUnit(TInputHandle oriHandle, TOutputHandle outputHandle)
        : m_inputHandle(std::move(oriHandle))
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        const auto& in = m_inputHandle.Data();
        m_outputHandle.Allocate(in.Shape());
        auto& out = m_outputHandle.MutableData();
        
        using ElementType = ElementTypePicker<decltype(out)>;
        
        const size_t count = in.Shape().Count();
        assert(count == out.Shape().Count());
        
        auto low_in = LowerAccess(in);
        ElementType* mem_in = low_in.MutableRawMemory();

        auto low_out = LowerAccess(out);
        ElementType* mem_out = low_out.MutableRawMemory();
                
        static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
        
        for (size_t i = 0; i < count; ++i)
        {
            mem_out[i] = (ElementType)(1 / (1 + exp(-mem_in[i])));
        }
        m_outputHandle.SetEval();
    }
    
private:
    const TInputHandle m_inputHandle;
    TOutputHandle m_outputHandle;
};
}

template <>
struct OperSeq_<OpTags::Sigmoid>
{
    using type = OperSeqContainer<TailCalculator<OperSigmoid::NSCaseGen::EvalUnit>>;
};

template <typename TP,
          typename = std::enable_if_t<IsValidOper<OpTags::Sigmoid, TP>>>
auto Sigmoid(TP&& p_m)
{
    using rawM = RemConstRef<TP>;
    using ResType = Operator<OpTags::Sigmoid, rawM>;
    return ResType(std::forward<TP>(p_m));
}
}