#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <MetaNN/facilities/_.h>
#include <MetaNN/operation/facilities/_.h>
#include <stdexcept>

namespace MetaNN::OpTags
{
    struct Divide;
    struct DivideFromNum;
    struct DivideByNum;
}

namespace MetaNN
{
// General Divide
namespace OperDivide::NSCaseGen
{
    template <typename TInputHandle1, typename TInputHandle2, typename TOutputHandle>
    class EvalItem : public BaseEvalItem
    {
        using CategoryTag = CategoryTagFromHandle<TOutputHandle>;
    public:
        EvalItem(TInputHandle1 oriHandle1, TInputHandle2 oriHandle2, TOutputHandle outputHandle,
                 Shape<CategoryTag::DimNum> shape)
            : BaseEvalItem(TypeID<EvalItem>(),
                           {oriHandle1.DataPtr(), oriHandle2.DataPtr()}, outputHandle.DataPtr())
            , m_inputHandle1(std::move(oriHandle1))
            , m_inputHandle2(std::move(oriHandle2))
            , m_outputHandle(std::move(outputHandle))
            , m_outputShape(std::move(shape))
        {}
        
        const TInputHandle1 m_inputHandle1;
        const TInputHandle2 m_inputHandle2;
        TOutputHandle m_outputHandle;
        Shape<CategoryTag::DimNum> m_outputShape;
    };

    template <typename TInputHandle1, typename TInputHandle2, typename TOutputHandle>
    class EvalGroup : public TrivialEvalGroup<EvalItem<TInputHandle1, TInputHandle2, TOutputHandle>>
    {
        using EvalItemType = EvalItem<TInputHandle1, TInputHandle2, TOutputHandle>;
    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            const auto& in1 = evalItem.m_inputHandle1.Data();
            const auto& in2 = evalItem.m_inputHandle2.Data();

            using ResType = typename TOutputHandle::DataType;
            using ElementType = typename ResType::ElementType;
            ResType out(in1.Shape());

            const size_t count1 = in1.Shape().Count();
            const size_t count2 = in2.Shape().Count();
            const size_t outCount = evalItem.m_outputShape.Count();
            assert(outCount % count1 == 0);
            assert(outCount % count2 == 0);

            auto low_in1 = LowerAccess(in1);
            const ElementType* mem_in1 = low_in1.RawMemory();
            auto low_in2 = LowerAccess(in2);
            const ElementType* mem_in2 = low_in2.RawMemory();

            auto low_out = LowerAccess(out);
            ElementType* mem_out = low_out.MutableRawMemory();

            static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");

            for (size_t i = 0; i < outCount; ++i)
            {
                mem_out[i] = mem_in1[i % count1] / mem_in2[i % count2];
            }
            evalItem.m_outputHandle.SetData(std::move(out));
        }
    };
}

template <>
struct OperSeq_<OpTags::Divide>
{
    using type = OperCalAlgoChain<TailCalculator<OperDivide::NSCaseGen::EvalItem,
                                                 OperDivide::NSCaseGen::EvalGroup,
                                                 PolicyContainer<PPassShape>>>;
};

/// Divide from number
namespace OperDivideFromNum
{
    template <typename TNumber, typename TOper>
    constexpr bool Valid()
    {
        if constexpr ((!IsValidCategoryTag<DataCategory<TOper>>) || (IsValidCategoryTag<DataCategory<TNumber>>))
        {
            return false;
        }
        else
        {
            return std::is_constructible_v<typename RemConstRef<TOper>::ElementType, TNumber>;
        }
    }
}
namespace OperDivideFromNum::NSCaseGen
{
    template <typename TInputHandle, typename TOutputHandle>
    class EvalItem : public BaseEvalItem
    {
        using CategoryTag = CategoryTagFromHandle<TOutputHandle>;
    public:
        template <typename TAuxParams>
        EvalItem(TInputHandle oriHandle, TOutputHandle outputHandle, const TAuxParams& params)
            : BaseEvalItem(TypeID<EvalItem>(),
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
    class EvalGroup : public TrivialEvalGroup<EvalItem<TInputHandle, TOutputHandle>>
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
                mem_out[i] = static_cast<ElementType>(evalItem.m_value) / mem_in[i];
            }
            evalItem.m_outputHandle.SetData(std::move(out));
        }
    };
}

template <typename TNumber, typename TOper>
constexpr bool IsValidOper<OpTags::DivideFromNum, TNumber, TOper>
    = OperDivideFromNum::Valid<TNumber, TOper>();

template <typename TElem, typename TCate>
struct OperAuxParams<OpTags::DivideFromNum, TElem, TCate> : public OperAuxValue<TElem>
{
    using TBase = OperAuxValue<TElem>;
    using TBase::TBase;
    using TBase::operator =;
};

template <>
struct OperSeq_<OpTags::DivideFromNum>
{
    using type = OperCalAlgoChain<TailCalculator<OperDivideFromNum::NSCaseGen::EvalItem,
                                                 OperDivideFromNum::NSCaseGen::EvalGroup,
                                                 PolicyContainer<PPassAuxParam>>>;
};

/// Divide by number
namespace OperDivideByNum
{
    template <typename TOper, typename TNumber>
    constexpr bool Valid()
    {
        if constexpr ((!IsValidCategoryTag<DataCategory<TOper>>) || (IsValidCategoryTag<DataCategory<TNumber>>))
        {
            return false;
        }
        else
        {
            return std::is_constructible_v<typename RemConstRef<TOper>::ElementType, TNumber>;
        }
    }
}

namespace OperDivideByNum::NSCaseGen
{
    template <typename TInputHandle, typename TOutputHandle>
    class EvalItem : public BaseEvalItem
    {
        using CategoryTag = CategoryTagFromHandle<TOutputHandle>;
    public:
        template <typename TAuxParams>
        EvalItem(TInputHandle oriHandle, TOutputHandle outputHandle, const TAuxParams& params)
            : BaseEvalItem(TypeID<EvalItem>(),
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
    class EvalGroup : public TrivialEvalGroup<EvalItem<TInputHandle, TOutputHandle>>
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
                mem_out[i] = mem_in[i] / static_cast<ElementType>(evalItem.m_value);
            }
            evalItem.m_outputHandle.SetData(std::move(out));
        }
    };
}

template <typename TOper, typename TNumber>
constexpr bool IsValidOper<OpTags::DivideByNum, TOper, TNumber>
    = OperDivideByNum::Valid<TOper, TNumber>();

template <typename TElem, typename TCate>
struct OperAuxParams<OpTags::DivideByNum, TElem, TCate> : public OperAuxValue<TElem>
{
    using TBase = OperAuxValue<TElem>;
    using TBase::TBase;
    using TBase::operator =;
};

template <>
struct OperSeq_<OpTags::DivideByNum>
{
    using type = OperCalAlgoChain<TailCalculator<OperDivideByNum::NSCaseGen::EvalItem,
                                                 OperDivideByNum::NSCaseGen::EvalGroup,
                                                 PolicyContainer<PPassAuxParam>>>;
};

/// Interface
template <typename TP1, typename TP2,
          std::enable_if_t<IsValidOper<OpTags::Divide, TP1, TP2> ||
                           IsValidOper<OpTags::DivideFromNum, TP1, TP2> ||
                           IsValidOper<OpTags::DivideByNum, TP1, TP2>>* = nullptr>
auto operator/ (TP1&& p_m1, TP2&& p_m2)
{
    if constexpr (IsValidOper<OpTags::Divide, TP1, TP2>)
    {
        using rawOp1 = RemConstRef<TP1>;
        using rawOp2 = RemConstRef<TP2>;
        using ResType = Operation<OpTags::Divide, OperandContainer<rawOp1, rawOp2>>;
        return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
    }
    else if constexpr (IsValidOper<OpTags::DivideFromNum, TP1, TP2>)
    {
        using rawOp = RemConstRef<TP2>;
        using ResType = Operation<OpTags::DivideFromNum, OperandContainer<rawOp>>;
        OperAuxParams<OpTags::DivideFromNum, typename rawOp::ElementType,
                      OperCateCal<OpTags::DivideFromNum, PolicyContainer<>, rawOp>> params(p_m1);
        return ResType(std::move(params), std::forward<TP2>(p_m2));
    }
    else if constexpr (IsValidOper<OpTags::DivideByNum, TP1, TP2>)
    {
        using rawOp = RemConstRef<TP1>;
        using ResType = Operation<OpTags::DivideByNum, OperandContainer<rawOp>>;
        OperAuxParams<OpTags::DivideByNum, typename rawOp::ElementType,
                      OperCateCal<OpTags::DivideByNum, PolicyContainer<>, rawOp>> params(p_m2);
        return ResType(std::move(params), std::forward<TP1>(p_m1));
    }
    else
    {
        static_assert(DependencyFalse<TP1>);
    }
}
}