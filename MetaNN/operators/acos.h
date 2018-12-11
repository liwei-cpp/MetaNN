#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <MetaNN/operators/facilities/tags.h>
#include <MetaNN/operators/facilities/operator_frame.h>
#include <cassert>
#include <cmath>
#include <type_traits>

namespace MetaNN
{
namespace OperAcos
{
template <typename TOperand>
static constexpr bool valid = !IsInvalid<TOperand>;

template <typename T>
static auto CreateOpTemplate(T&& p_m)
{
    using rawM = RemConstRef<T>;
    using ResType = Operator<OpTags::Acos, rawM>;
    return ResType(std::forward<T>(p_m));
}

namespace NSCaseGen
{
template <typename TInputHandle, typename TOutputHandle, typename TDevice>
class EvalUnit : public BaseEvalUnit<TDevice>
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
                
        static_assert(std::is_same_v<TDevice, DeviceTags::CPU>, "Currently only CPU is supported");
        
        for (size_t i = 0; i < count; ++i)
        {
            mem_out[i] = std::acos(mem_in[i]);
        }
        m_outputHandle.SetEval();
    }
    
private:
    const TInputHandle m_inputHandle;
    TOutputHandle m_outputHandle;
};

struct Calculator
{
    template <typename TCaseTail, typename TEvalRes, typename TOp>
    static void EvalRegister(TEvalRes& evalRes, const TOp& oper)
    {
        static_assert(std::is_same_v<TCaseTail, OperSeqContainer<>>,
                      "General case is not the last one");
                      
        using DeviceType = typename TEvalRes::DataType::DeviceType;

        const auto& data = oper.template GetOperand<0>();
        auto handle = data.EvalRegister();
        
        auto outHandle = evalRes.Handle();
        using UnitType = EvalUnit<decltype(handle), decltype(outHandle), DeviceType>;
        using GroupType = TrivalEvalGroup<UnitType>;

        const void* dataPtr = outHandle.DataPtr();
        const void* depVec = handle.DataPtr();
        UnitType unit(std::move(handle), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, {depVec});
    }
};
}
}

template <>
struct OperSeq_<OpTags::Acos>
{
    using type = OperSeqContainer<OperAcos::NSCaseGen::Calculator>;
};

template <typename TP,
          typename = std::enable_if_t<OperAcos::valid<TP>>>
auto Acos(TP&& p_m)
{
    return OperAcos::CreateOpTemplate(std::forward<TP>(p_m));
}
}