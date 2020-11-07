#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <MetaNN/operation/facilities/_.h>
#include <MetaNN/facilities/_.h>
#include <cassert>
#include <type_traits>

namespace MetaNN::OpTags
{
    struct Abs;
}

namespace MetaNN
{
namespace OperAbs::NSCaseGen
{
    template <typename TInputHandle, typename TOutputHandle>
    class EvalItem : public BaseEvalItem
    {
        using CategoryTag = CategoryTagFromHandle<TOutputHandle>;
    public:
        EvalItem(TInputHandle oriHandle, TOutputHandle outputHandle)
            : BaseEvalItem(TypeID<EvalItem>(),
                           {oriHandle.DataPtr()}, outputHandle.DataPtr())
            , m_inputHandle(std::move(oriHandle))
            , m_outputHandle(std::move(outputHandle))
        {}
        
        const TInputHandle m_inputHandle;
        TOutputHandle m_outputHandle;
    };
    
    template <typename TInputHandle, typename TOutputHandle>
    class EvalGroup : public TrivialEvalGroup<EvalItem<TInputHandle, TOutputHandle>>
    {
        using EvalItemType = EvalItem<TInputHandle, TOutputHandle>;
    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            const auto& in = evalItem.m_inputHandle.Data();

            using ResType = typename TOutputHandle::DataType;
            using ElementType = typename ResType::ElementType;
            ResType out(in.Shape());

            const size_t count = in.Shape().Count();
            assert(count == out.Shape().Count());

            auto low_in = LowerAccess(in);
            const ElementType* mem_in = low_in.RawMemory();

            auto low_out = LowerAccess(out);
            ElementType* mem_out = low_out.MutableRawMemory();

            static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");

            const ElementType zero{};
            for (size_t i = 0; i < count; ++i)
            {
                const ElementType val = mem_in[i];
                mem_out[i] = (val > zero) ? val : -val;
            }
            evalItem.m_outputHandle.SetData(std::move(out));
        }
    };
}

template <>
struct OperSeq_<OpTags::Abs>
{
    using type = OperCalAlgoChain<TailCalculator<OperAbs::NSCaseGen::EvalItem, OperAbs::NSCaseGen::EvalGroup>>;
};

template <typename TP,
          std::enable_if_t<IsValidOper<OpTags::Abs, TP>>* = nullptr>
auto Abs(TP&& p_m)
{
    using rawM = RemConstRef<TP>;
    using ResType = Operation<OpTags::Abs, OperandContainer<rawM>>;
    return ResType(std::forward<TP>(p_m));
}
}