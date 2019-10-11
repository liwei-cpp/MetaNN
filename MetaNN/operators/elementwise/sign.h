#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <MetaNN/operators/facilities/operator_frame.h>
#include <cassert>
#include <type_traits>

namespace MetaNN::OpTags
{
    struct Sign;
}

namespace MetaNN
{
namespace OperSign::NSCaseGen
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

        using ResType = typename TOutputHandle::DataType;
        using ElementType = typename ResType::ElementType;
        ResType out(in.Shape());

        const size_t count = in.Shape().Count();
        assert(count == out.Shape().Count());
        
        auto low_in = LowerAccess(in);
        ElementType* mem_in = low_in.MutableRawMemory();

        auto low_out = LowerAccess(out);
        ElementType* mem_out = low_out.MutableRawMemory();
                
        static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
        
        const ElementType zero{};
        const ElementType one{1};
        const ElementType negOne{-1};
        for (size_t i = 0; i < count; ++i)
        {
            if (mem_in[i] == zero)
                mem_out[i] = zero;
            else
                mem_out[i] = (mem_in[i] > zero) ? one : negOne;
        }
        m_outputHandle.SetData(std::move(out));
    }
    
private:
    const TInputHandle m_inputHandle;
    TOutputHandle m_outputHandle;
};
}

template <>
struct OperSeq_<OpTags::Sign>
{
    using type = OperSeqContainer<TailCalculator<OperSign::NSCaseGen::EvalUnit>>;
};

template <typename TP,
          typename = std::enable_if_t<IsValidOper<OpTags::Sign, TP>>>
auto Sign(TP&& p_m)
{
    using rawM = RemConstRef<TP>;
    using ResType = Operator<OpTags::Sign, rawM>;
    return ResType(std::forward<TP>(p_m));

}
}