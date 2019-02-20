#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <MetaNN/operators/facilities/tail_calculator.h>
#include <stdexcept>

namespace MetaNN::OpTags
{
    struct Add;
    struct AddWithNum;
}

namespace MetaNN
{
namespace OperAdd::NSCaseGen
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
                
        static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
        
        for (size_t i = 0; i < count; ++i)
        {
            mem_out[i] = mem_in1[i] + mem_in2[i];
        }
        m_outputHandle.SetEval();
    }
    
private:
    const TInputHandle1 m_inputHandle1;
    const TInputHandle2 m_inputHandle2;
    TOutputHandle m_outputHandle;
};
}

template <>
struct OperSeq_<OpTags::Add>
{
    using type = OperSeqContainer<TailCalculator<OperAdd::NSCaseGen::EvalUnit>>;
};

// add with number
namespace OperAddWithNum
{
template <typename TOp1, typename TOp2>
constexpr bool Valid()
{
    if constexpr (IsInvalid<TOp1> && !IsInvalid<TOp2>)
    {
        return std::is_constructible_v<typename RemConstRef<TOp2>::ElementType, TOp1>;
    }
    else if constexpr (!IsInvalid<TOp1> && IsInvalid<TOp2>)
    {
        return std::is_constructible_v<typename RemConstRef<TOp1>::ElementType, TOp2>;
    }
    else
    {
        return false;
    }
}

namespace NSCaseGen
{
template <typename TInputHandle, typename TOutputHandle>
class EvalUnit : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
public:
    template <typename TAuxParams>
    EvalUnit(TInputHandle oriHandle, TOutputHandle outputHandle, const TAuxParams& params)
        : m_inputHandle(std::move(oriHandle))
        , m_value(params.Value())
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        const auto& input = m_inputHandle.Data();
        
        m_outputHandle.Allocate(input.Shape());
        auto& out = m_outputHandle.MutableData();
        
        using ElementType = ElementTypePicker<decltype(out)>;
        
        const size_t count = input.Shape().Count();
        assert(count == out.Shape().Count());
        
        auto low_in = LowerAccess(input);
        ElementType* mem_in = low_in.MutableRawMemory();

        auto low_out = LowerAccess(out);
        ElementType* mem_out = low_out.MutableRawMemory();
                
        static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
        
        for (size_t i = 0; i < count; ++i)
        {
            mem_out[i] = mem_in[i] + m_value;
        }
        m_outputHandle.SetEval();
    }
    
private:
    const TInputHandle m_inputHandle;
    double m_value;
    TOutputHandle m_outputHandle;
};
}
}

template <typename TOp1, typename TOp2>
constexpr bool IsValidOper<OpTags::AddWithNum, TOp1, TOp2> = OperAddWithNum::Valid<TOp1, TOp2>();

template <typename TCate>
struct OperAuxParams<OpTags::AddWithNum, TCate> : public OperAuxValue<double>
{
    using TBase = OperAuxValue<double>;
    using TBase::TBase;
    using TBase::operator =;
};

template <>
struct OperSeq_<OpTags::AddWithNum>
{
    using type = OperSeqContainer<TailCalculator<OperAddWithNum::NSCaseGen::EvalUnit>>;
};

// interface
template <typename TP1, typename TP2,
          typename = std::enable_if_t<IsValidOper<OpTags::Add, TP1, TP2> ||
                                      IsValidOper<OpTags::AddWithNum, TP1, TP2>>>
auto operator+ (TP1&& p_m1, TP2&& p_m2)
{
    if constexpr (IsValidOper<OpTags::Add, TP1, TP2>)
    {
        if (p_m1.Shape() != p_m2.Shape())
        {
            throw std::runtime_error("Add error: operands' shape mismatch.");
        }
    
        using rawOp1 = RemConstRef<TP1>;
        using rawOp2 = RemConstRef<TP2>;
        using ResType = Operator<OpTags::Add, rawOp1, rawOp2>;
        return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
    }
    else if constexpr (IsValidOper<OpTags::AddWithNum, TP1, TP2>)
    {
        if constexpr (IsInvalid<TP1> && !IsInvalid<TP2>)
        {
            using rawOp = RemConstRef<TP2>;
            using ResType = Operator<OpTags::AddWithNum, rawOp>;
            OperAuxParams<OpTags::AddWithNum, OperCateCal<OpTags::AddWithNum, rawOp>> params(p_m1);
            return ResType(std::move(params), std::forward<TP2>(p_m2));
        }
        else if constexpr (!IsInvalid<TP1> && IsInvalid<TP2>)
        {
            using rawOp = RemConstRef<TP1>;
            using ResType = Operator<OpTags::AddWithNum, rawOp>;
            OperAuxParams<OpTags::AddWithNum, OperCateCal<OpTags::AddWithNum, rawOp>> params(p_m2);
            return ResType(std::move(params), std::forward<TP1>(p_m1));
        }
        else
        {
            static_assert(DependencyFalse<TP1, TP2>);
        }
    }
    else
    {
        static_assert(DependencyFalse<TP1, TP2>);
    }
}
}