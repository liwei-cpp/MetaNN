#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <MetaNN/operators/facilities/tags.h>
#include <MetaNN/operators/facilities/operator_frame.h>
#include <stdexcept>

namespace MetaNN
{
namespace OperatorMultiply
{
template <typename TOp1, typename TOp2>
static constexpr bool valid = (!IsInvalid<TOp1>) &&
                              (!IsInvalid<TOp2>) &&
                              (std::is_same_v<DataCategory<TOp1>, DataCategory<TOp2>>);

template <typename TOp1, typename TOp2>
static auto CreateOpTemplate(TOp1&& p1, TOp2&& p2)
{
    if (p1.Shape() != p2.Shape())
    {
        throw std::runtime_error("Multiply error: operands' shape mismatch.");
    }
    
    using rawOp1 = RemConstRef<TOp1>;
    using rawOp2 = RemConstRef<TOp2>;
    using ResType = Operator<OpTags::Multiply, rawOp1, rawOp2>;
    return ResType(std::forward<TOp1>(p1), std::forward<TOp2>(p2));
}

namespace NSCaseGen
{
template <typename TInputHandle1, typename TInputHandle2, typename TOutputHandle, typename TDevice>
class EvalUnit : public BaseEvalUnit<TDevice>
{
public:
    EvalUnit(TInputHandle1 oriHandle1, TInputHandle2 oriHandle2, TOutputHandle outputHandle)
        : m_inputHandle1(std::move(oriHandle1))
        , m_inputHandle2(std::move(oriHandle2))
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        const auto& in1 = m_inputHandle1.Data();
        const auto& in2 = m_inputHandle2.Data();
        assert(in1.Shape() == in2.Shape());
        
        m_outputHandle.Allocate(in1.Shape());
        auto& out = m_outputHandle.MutableData();
        
        using ElementType = ElementTypePicker<decltype(out)>;
        
        const size_t count = in1.Shape().Count();
        assert(count == out.Shape().Count());
        
        auto low_in1 = LowerAccess(in1);
        ElementType* mem_in1 = low_in1.MutableRawMemory();
        auto low_in2 = LowerAccess(in2);
        ElementType* mem_in2 = low_in2.MutableRawMemory();

        auto low_out = LowerAccess(out);
        ElementType* mem_out = low_out.MutableRawMemory();
                
        static_assert(std::is_same_v<TDevice, DeviceTags::CPU>, "Currently only CPU is supported");
        
        for (size_t i = 0; i < count; ++i)
        {
            mem_out[i] = mem_in1[i] * mem_in2[i];
        }
        m_outputHandle.SetEval();
    }
    
private:
    const TInputHandle1 m_inputHandle1;
    const TInputHandle2 m_inputHandle2;
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

        const auto& data1 = oper.template GetOperand<0>();
        const auto& data2 = oper.template GetOperand<1>();
        auto handle1 = data1.EvalRegister();
        auto handle2 = data2.EvalRegister();
        
        auto outHandle = evalRes.Handle();
        using UnitType = EvalUnit<decltype(handle1), decltype(handle2), decltype(outHandle), DeviceType>;
        using GroupType = TrivalEvalGroup<UnitType>;

        const void* dataPtr = outHandle.DataPtr();
        std::vector<const void*> depVec{handle1.DataPtr(), handle2.DataPtr()};
        UnitType unit(std::move(handle1), std::move(handle2), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, depVec);
    }
};
}
}

template <>
struct OperSeq_<OpTags::Multiply>
{
    using type = OperSeqContainer<OperatorMultiply::NSCaseGen::Calculator>;
};

template <typename TP1, typename TP2,
          typename = std::enable_if_t<OperatorMultiply::valid<TP1, TP2>>>
auto operator* (TP1&& p_m1, TP2&& p_m2)
{
    return OperatorMultiply::CreateOpTemplate(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
}
}