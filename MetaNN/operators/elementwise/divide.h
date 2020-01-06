#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_plan.h>
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
    class EvalItem : public BaseEvalItem<DeviceTypeFromHandle<TOutputHandle>>
    {
        using BaseType = BaseEvalItem<DeviceTypeFromHandle<TOutputHandle>>;
    public:
        template <typename TAuxParams>
        EvalItem(TInputHandle1 oriHandle1, TInputHandle2 oriHandle2, TOutputHandle outputHandle, const TAuxParams&)
            : BaseType(std::type_index(typeid(EvalItem)),
                       {oriHandle1.DataPtr(), oriHandle2.DataPtr()}, outputHandle.DataPtr())
            , m_inputHandle1(std::move(oriHandle1))
            , m_inputHandle2(std::move(oriHandle2))
            , m_outputHandle(std::move(outputHandle))
        {}
        
        const TInputHandle1 m_inputHandle1;
        const TInputHandle2 m_inputHandle2;
        TOutputHandle m_outputHandle;
    };

    template <typename TInputHandle1, typename TInputHandle2, typename TOutputHandle>
    class EvalGroup : public TrivalEvalGroup<EvalItem<TInputHandle1, TInputHandle2, TOutputHandle>>
    {
        using EvalItemType = EvalItem<TInputHandle1, TInputHandle2, TOutputHandle>;
    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            const auto& in1 = evalItem.m_inputHandle1.Data();
            const auto& in2 = evalItem.m_inputHandle2.Data();
            assert(in1.Shape() == in2.Shape());

            using ResType = typename TOutputHandle::DataType;
            using ElementType = typename ResType::ElementType;
            ResType out(in1.Shape());

            const size_t count = in1.Shape().Count();
            assert(count == out.Shape().Count());

            auto low_in1 = LowerAccess(in1);
            const ElementType* mem_in1 = low_in1.RawMemory();
            auto low_in2 = LowerAccess(in2);
            const ElementType* mem_in2 = low_in2.RawMemory();

            auto low_out = LowerAccess(out);
            ElementType* mem_out = low_out.MutableRawMemory();

            static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");

            for (size_t i = 0; i < count; ++i)
            {
                mem_out[i] = mem_in1[i] / mem_in2[i];
            }
            evalItem.m_outputHandle.SetData(std::move(out));
        }
    };
}

template <>
struct OperSeq_<OpTags::Divide>
{
    using type = OperCalAlgoChain<TailCalculator<OperDivide::NSCaseGen::EvalItem, OperDivide::NSCaseGen::EvalGroup>>;
};

/// Divide by number
namespace OperDivideByNum::NSCaseGen
{
    template <typename TInputHandle, typename TOutputHandle>
    class EvalItem : public BaseEvalItem<DeviceTypeFromHandle<TOutputHandle>>
    {
        using BaseType = BaseEvalItem<DeviceTypeFromHandle<TOutputHandle>>;
    public:
        template <typename TAuxParams>
        EvalItem(TInputHandle oriHandle, TOutputHandle outputHandle, const TAuxParams& params)
            : BaseType(std::type_index(typeid(EvalItem)),
                       {oriHandle.DataPtr()}, outputHandle.DataPtr())
            , m_inputHandle(std::move(oriHandle))
            , m_value(params.Value())
            , m_outputHandle(std::move(outputHandle))
        {}
        
        const TInputHandle m_inputHandle;
        double m_value;
        TOutputHandle m_outputHandle;
    };

    template <typename TInputHandle, typename TOutputHandle>
    class EvalGroup : public TrivalEvalGroup<EvalItem<TInputHandle, TOutputHandle>>
    {
        using EvalItemType = EvalItem<TInputHandle, TOutputHandle>;
    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            const auto& input = evalItem.m_inputHandle.Data();

            using ResType = typename TOutputHandle::DataType;
            using ElementType = typename ResType::ElementType;
            ResType out(input.Shape());

            const size_t count = input.Shape().Count();
            assert(count == out.Shape().Count());

            auto low_in = LowerAccess(input);
            const ElementType* mem_in = low_in.RawMemory();

            auto low_out = LowerAccess(out);
            ElementType* mem_out = low_out.MutableRawMemory();

            static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");

            for (size_t i = 0; i < count; ++i)
            {
                mem_out[i] = mem_in[i] / evalItem.m_value;
            }
            evalItem.m_outputHandle.SetData(std::move(out));
        }
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
    using type = OperCalAlgoChain<TailCalculator<OperDivideByNum::NSCaseGen::EvalItem, OperDivideByNum::NSCaseGen::EvalGroup>>;
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