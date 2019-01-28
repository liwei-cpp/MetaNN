#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <MetaNN/operators/facilities/tail_calculator.h>
#include <MetaNN/operators/loss/facilities/organizer.h>
#include <MetaNN/operators/loss/facilities/tags.h>
#include <cassert>
#include <cmath>
#include <type_traits>

namespace MetaNN
{
namespace OperNLLLoss::NSCaseGen
{
template <typename TWeightHandle, typename TInputHandle, typename TOutputHandle>
class EvalUnit : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
public:
    template <typename TAuxParams>
    EvalUnit(TWeightHandle weightHandle, TInputHandle inputHandle, TOutputHandle outputHandle, const TAuxParams&)
        : m_weightHandle(std::move(weightHandle))
        , m_inputHandle(std::move(inputHandle))
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        const auto& weight = m_weightHandle.Data();
        const auto& in = m_inputHandle.Data();
        assert(weight.Shape() == in.Shape());
        
        m_outputHandle.Allocate();
        auto& out = m_outputHandle.MutableData();
        
        using ElementType = ElementTypePicker<decltype(out)>;
        
        const size_t inCount = in.Shape().Count();
        
        auto low_in = LowerAccess(in);
        ElementType* mem_in = low_in.MutableRawMemory();

        auto low_weight = LowerAccess(weight);
        ElementType* mem_weight = low_weight.MutableRawMemory();
                
        static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
        
        ElementType res{};
        for (size_t i = 0; i < inCount; ++i)
        {
            res -= mem_weight[i] * log(mem_in[i]);
        }
        
        if constexpr (!IsCardinal<decltype(weight)>)
        {
            const size_t cardinalCount = in.Shape().Cardinal().Count();
            assert(inCount % cardinalCount == 0);
            res /= (inCount / cardinalCount);
        }
        out.SetValue(res);
        m_outputHandle.SetEval();
    }
    
private:
    const TWeightHandle m_weightHandle;
    const TInputHandle m_inputHandle;
    TOutputHandle m_outputHandle;
};
}

template <typename TWeight, typename TInput>
struct OperCategory_<OpTags::NLLLoss, TWeight, TInput>
    : public GenLossOperCategory_
{};

template <>
class OperShapeInfo<OpTags::NLLLoss, CategoryTags::Scalar>
    : public GenLossOperShapeInfo
{
public:
    using GenLossOperShapeInfo::GenLossOperShapeInfo;
};

template <>
struct OperSeq_<OpTags::NLLLoss>
{
    using type = OperSeqContainer<TailCalculator<OperNLLLoss::NSCaseGen::EvalUnit>>;
};

template <typename TWeight, typename TInput,
          typename = std::enable_if_t<IsValidOper<OpTags::NLLLoss, TWeight, TInput>>>
auto NLLLoss(TWeight&& p_weight, TInput&& p_input)
{
    using ResType = Operator<OpTags::NLLLoss, RemConstRef<TWeight>, RemConstRef<TInput>>;
    return ResType(std::forward<TWeight>(p_weight), std::forward<TInput>(p_input));
}
}