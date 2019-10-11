#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <MetaNN/operators/facilities/tail_calculator.h>
#include <stdexcept>

namespace MetaNN::OpTags
{
    struct Interpolate;
}

namespace MetaNN
{
namespace OperInterpolate::NSCaseGen
{
template <typename TInputHandle1, typename TInputHandle2, typename TInputHandle3,
          typename TOutputHandle>
class EvalUnit : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
public:
    template <typename TAuxParams>
    EvalUnit(TInputHandle1 oriHandle1, TInputHandle2 oriHandle2, TInputHandle3 oriHandle3,
             TOutputHandle outputHandle, const TAuxParams&)
        : m_inputHandle1(std::move(oriHandle1))
        , m_inputHandle2(std::move(oriHandle2))
        , m_inputHandle3(std::move(oriHandle3))
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        const auto& in1 = m_inputHandle1.Data();
        const auto& in2 = m_inputHandle2.Data();
        const auto& in3 = m_inputHandle3.Data();
        assert(in1.Shape() == in2.Shape());
        assert(in1.Shape() == in3.Shape());
        
        using ResType = typename TOutputHandle::DataType;
        using ElementType = typename ResType::ElementType;
        ResType out(in1.Shape());

        const size_t count = in1.Shape().Count();
        assert(count == out.Shape().Count());
        
        auto low_in1 = LowerAccess(in1);
        ElementType* mem_in1 = low_in1.MutableRawMemory();
        auto low_in2 = LowerAccess(in2);
        ElementType* mem_in2 = low_in2.MutableRawMemory();
        auto low_in3 = LowerAccess(in3);
        ElementType* mem_in3 = low_in3.MutableRawMemory();

        auto low_out = LowerAccess(out);
        ElementType* mem_out = low_out.MutableRawMemory();
                
        static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
        
        for (size_t i = 0; i < count; ++i)
        {
            mem_out[i] = mem_in1[i] * mem_in3[i] + mem_in2[i] * (1 - mem_in3[i]);
        }
        m_outputHandle.SetData(std::move(out));
    }
    
private:
    const TInputHandle1 m_inputHandle1;
    const TInputHandle2 m_inputHandle2;
    const TInputHandle3 m_inputHandle3;
    TOutputHandle m_outputHandle;
};
}

template <>
struct OperSeq_<OpTags::Interpolate>
{
    using type = OperSeqContainer<TailCalculator<OperInterpolate::NSCaseGen::EvalUnit>>;
};

template <typename TP1, typename TP2, typename TP3,
          typename = std::enable_if_t<IsValidOper<OpTags::Interpolate, TP1, TP2, TP3>>>
auto Interpolate(TP1&& p_m1, TP2&& p_m2, TP3&& p_m3)
{
    if ((p_m1.Shape() != p_m2.Shape()) || (p_m1.Shape() != p_m3.Shape()))
    {
        throw std::runtime_error("Interpolate error: operands' shape mismatch.");
    }
    using ResType = Operator<OpTags::Interpolate,
                             RemConstRef<TP1>, RemConstRef<TP2>, RemConstRef<TP3>>;
    return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2), std::forward<TP3>(p_m3));
}
}