#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <MetaNN/facilities/_.h>
#include <MetaNN/operation/facilities/_.h>
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
    class EvalItem : public BaseEvalItem
    {
        using CategoryTag = CategoryTagFromHandle<TOutputHandle>;
    public:
        EvalItem(TInputHandle1 oriHandle1, TInputHandle2 oriHandle2, TInputHandle3 oriHandle3,
                 TOutputHandle outputHandle, Shape<CategoryTag::DimNum> shape)
            : BaseEvalItem(TypeID<EvalItem>(),
                           {oriHandle1.DataPtr(), oriHandle2.DataPtr(), oriHandle3.DataPtr()},
                           outputHandle.DataPtr())
            , m_inputHandle1(std::move(oriHandle1))
            , m_inputHandle2(std::move(oriHandle2))
            , m_inputHandle3(std::move(oriHandle3))
            , m_outputHandle(std::move(outputHandle))
            , m_outputShape(std::move(shape))
        {}
        
        const TInputHandle1 m_inputHandle1;
        const TInputHandle2 m_inputHandle2;
        const TInputHandle3 m_inputHandle3;
        TOutputHandle m_outputHandle;
        Shape<CategoryTag::DimNum> m_outputShape;
    };
    
    template <typename TInputHandle1, typename TInputHandle2, typename TInputHandle3,
              typename TOutputHandle>
    class EvalGroup : public TrivialEvalGroup<EvalItem<TInputHandle1, TInputHandle2, TInputHandle3, TOutputHandle>>
    {
        using EvalItemType = EvalItem<TInputHandle1, TInputHandle2, TInputHandle3, TOutputHandle>;
    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            const auto& in1 = evalItem.m_inputHandle1.Data();
            const auto& in2 = evalItem.m_inputHandle2.Data();
            const auto& in3 = evalItem.m_inputHandle3.Data();

            using ResType = typename TOutputHandle::DataType;
            using ElementType = typename ResType::ElementType;
            ResType out(evalItem.m_outputShape);

            const size_t count1 = in1.Shape().Count();
            const size_t count2 = in2.Shape().Count();
            const size_t count3 = in3.Shape().Count();
            const size_t outCount = evalItem.m_outputShape.Count();
            assert(outCount % count1 == 0);
            assert(outCount % count2 == 0);
            assert(outCount % count3 == 0);

            auto low_in1 = LowerAccess(in1);
            const ElementType* mem_in1 = low_in1.RawMemory();
            auto low_in2 = LowerAccess(in2);
            const ElementType* mem_in2 = low_in2.RawMemory();
            auto low_in3 = LowerAccess(in3);
            const ElementType* mem_in3 = low_in3.RawMemory();

            auto low_out = LowerAccess(out);
            ElementType* mem_out = low_out.MutableRawMemory();

            static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");

            for (size_t i = 0; i < outCount; ++i)
            {
                mem_out[i] = mem_in1[i % count1] * mem_in3[i % count3] + mem_in2[i % count2] * (1 - mem_in3[i % count3]);
            }
            evalItem.m_outputHandle.SetData(std::move(out));
        }
    };
}

template <>
struct OperSeq_<OpTags::Interpolate>
{
    using type = OperCalAlgoChain<TailCalculator<OperInterpolate::NSCaseGen::EvalItem,
                                                 OperInterpolate::NSCaseGen::EvalGroup,
                                                 PolicyContainer<PPassShape>>>;
};

template <typename TP1, typename TP2, typename TP3,
          std::enable_if_t<IsValidOper<OpTags::Interpolate, TP1, TP2, TP3>>* = nullptr>
auto Interpolate(TP1&& p_m1, TP2&& p_m2, TP3&& p_m3)
{
    using ResType = Operation<OpTags::Interpolate,
                              OperandContainer<RemConstRef<TP1>, RemConstRef<TP2>, RemConstRef<TP3>>>;
    return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2), std::forward<TP3>(p_m3));
}
}