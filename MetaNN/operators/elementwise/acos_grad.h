#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <MetaNN/operators/elementwise/tags.h>
#include <MetaNN/operators/facilities/operator_frame.h>
#include <cassert>
#include <cmath>
#include <type_traits>

namespace MetaNN
{
namespace OperAcosGrad::NSCaseGen
{
template <typename TGradHandle, typename TInputHandle, typename TOutputHandle>
class EvalUnit : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
public:
    template <typename TAuxParams>
    EvalUnit(TGradHandle gradHandle, TInputHandle oriHandle, TOutputHandle outputHandle, const TAuxParams&)
        : m_gradHandle(std::move(gradHandle))
        , m_inputHandle(std::move(oriHandle))
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        const auto& grad = m_gradHandle.Data();
        const auto& in = m_inputHandle.Data();
        assert(grad.Shape() == in.Shape());
        
        m_outputHandle.Allocate(in.Shape());
        auto& out = m_outputHandle.MutableData();
        
        using ElementType = ElementTypePicker<decltype(out)>;
        
        const size_t count = in.Shape().Count();
        assert(count == out.Shape().Count());

        auto low_grad = LowerAccess(grad);
        ElementType* mem_grad = low_grad.MutableRawMemory();
        auto low_in = LowerAccess(in);
        ElementType* mem_in = low_in.MutableRawMemory();

        auto low_out = LowerAccess(out);
        ElementType* mem_out = low_out.MutableRawMemory();
                
        static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
        
        for (size_t i = 0; i < count; ++i)
        {
            mem_out[i] = -mem_grad[i] / std::sqrt(1 - mem_in[i] * mem_in[i]);
        }
        m_outputHandle.SetEval();
    }
    
private:
    const TGradHandle m_gradHandle;
    const TInputHandle m_inputHandle;
    TOutputHandle m_outputHandle;
};
}

template <>
struct OperSeq_<OpTags::AcosGrad>
{
    using type = OperSeqContainer<TailCalculator<OperAcosGrad::NSCaseGen::EvalUnit>>;
};

template <typename TGrad, typename TInput,
          typename = std::enable_if_t<IsValidOper<OpTags::AcosGrad, TGrad, TInput>>>
auto AcosGrad(TGrad&& p_grad, TInput&& p_input)
{
    using rawGrad = RemConstRef<TGrad>;
    using rawInput = RemConstRef<TInput>;
    using ResType = Operator<OpTags::AcosGrad, rawGrad, rawInput>;
    return ResType(std::forward<TGrad>(p_grad), std::forward<TInput>(p_input));
}
}