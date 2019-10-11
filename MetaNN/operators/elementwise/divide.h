#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <MetaNN/operators/facilities/instance_id.h>
#include <MetaNN/operators/facilities/tail_calculator.h>
#include <stdexcept>

namespace MetaNN::OpTags
{
    struct Divide;
    struct DivideByNum;
}

namespace MetaNN
{
// General Divide
namespace OperDivide::NSCaseGen
{
template <typename TInputHandle1, typename TInputHandle2, typename TOutputHandle>
class EvalUnit : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
public:
    template <typename TAuxParams>
    EvalUnit(TInputHandle1 oriHandle1, TInputHandle2 oriHandle2, TOutputHandle outputHandle, const TAuxParams&)
        : m_inputHandle1(std::move(oriHandle1))
        , m_inputHandle2(std::move(oriHandle2))
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        const auto& in1 = m_inputHandle1.Data();
        const auto& in2 = m_inputHandle2.Data();
        assert(in1.Shape() == in2.Shape());
        
        using ResType = typename TOutputHandle::DataType;
        using ElementType = typename ResType::ElementType;
        ResType out(in1.Shape());

        const size_t count = in1.Shape().Count();
        assert(count == out.Shape().Count());
        
        auto low_in1 = LowerAccess(in1);
        ElementType* mem_in1 = low_in1.MutableRawMemory();
        auto low_in2 = LowerAccess(in2);
        ElementType* mem_in2 = low_in2.MutableRawMemory();

        auto low_out = LowerAccess(out);
        ElementType* mem_out = low_out.MutableRawMemory();
                
        static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
        
        for (size_t i = 0; i < count; ++i)
        {
            mem_out[i] = mem_in1[i] / mem_in2[i];
        }
        m_outputHandle.SetData(std::move(out));
    }
    
private:
    const TInputHandle1 m_inputHandle1;
    const TInputHandle2 m_inputHandle2;
    TOutputHandle m_outputHandle;
};
}

template <>
struct OperSeq_<OpTags::Divide>
{
    using type = OperSeqContainer<TailCalculator<OperDivide::NSCaseGen::EvalUnit>>;
};

/// Divide by number
namespace OperDivideByNum::NSCaseGen
{
template <typename TInputHandle, typename TOutputHandle>
class EvalUnit : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
public:
    template <typename TAuxParams>
    EvalUnit(TInputHandle oriHandle, TOutputHandle outputHandle, const TAuxParams& params)
        : m_inputHandle(std::move(oriHandle))
        , m_dividend(params.Value())
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        const auto& input = m_inputHandle.Data();
        
        using ResType = typename TOutputHandle::DataType;
        using ElementType = typename ResType::ElementType;
        ResType out(input.Shape());

        const size_t count = input.Shape().Count();
        assert(count == out.Shape().Count());
        
        auto low_in = LowerAccess(input);
        ElementType* mem_in = low_in.MutableRawMemory();

        auto low_out = LowerAccess(out);
        ElementType* mem_out = low_out.MutableRawMemory();
                
        static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
        
        for (size_t i = 0; i < count; ++i)
        {
            mem_out[i] = mem_in[i] / m_dividend;
        }
        m_outputHandle.SetData(std::move(out));
    }
    
private:
    const TInputHandle m_inputHandle;
    double m_dividend;
    TOutputHandle m_outputHandle;
};
}

template <typename TOper, typename TNumber>
constexpr bool IsValidOper<OpTags::DivideByNum, TOper, TNumber>
    = (IsInDataCategory<TOper>) && (std::is_constructible_v<typename RemConstRef<TOper>::ElementType, TNumber>);

template <typename TCate>
struct OperAuxParams<OpTags::DivideByNum, TCate> : public OperAuxValue<double>
{
    using TBase = OperAuxValue<double>;
    using TBase::TBase;
    using TBase::operator =;
};

template <>
struct OperSeq_<OpTags::DivideByNum>
{
    using type = OperSeqContainer<TailCalculator<OperDivideByNum::NSCaseGen::EvalUnit>>;
};

/// Interface
template <typename TP1, typename TP2,
          typename = std::enable_if_t<IsValidOper<OpTags::Divide, TP1, TP2> ||
                                      IsValidOper<OpTags::DivideByNum, TP1, TP2>>>
auto operator/ (TP1&& p_m1, TP2&& p_m2)
{
    if constexpr (IsValidOper<OpTags::Divide, TP1, TP2>)
    {
        if (p_m1.Shape() != p_m2.Shape())
        {
            throw std::runtime_error("Divide error: operands' shape mismatch.");
        }
    
        using rawOp1 = RemConstRef<TP1>;
        using rawOp2 = RemConstRef<TP2>;
        using ResType = Operator<OpTags::Divide, rawOp1, rawOp2>;
        return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
    }
    else if constexpr (IsValidOper<OpTags::DivideByNum, TP1, TP2>)
    {
        using rawOp = RemConstRef<TP1>;
        using ResType = Operator<OpTags::DivideByNum, rawOp>;
        OperAuxParams<OpTags::DivideByNum, OperCateCal<OpTags::DivideByNum, rawOp>> params(p_m2);
        return ResType(std::move(params), std::forward<TP1>(p_m1));
    }
    else
    {
        static_assert(DependencyFalse<TP1>);
    }
}
}