#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <MetaNN/operators/facilities/tail_calculator.h>
#include <MetaNN/operators/loss/facilities/organizer.h>
#include <cassert>
#include <cmath>
#include <type_traits>

namespace MetaNN::OpTags
{
    struct NLLLoss;
    struct NLLLossGrad;
}

namespace MetaNN
{
namespace OperNLLLoss::NSCaseGen
{
    template <typename TWeightHandle, typename TInputHandle, typename TOutputHandle>
    class EvalItem : public BaseEvalItem<DeviceTypeFromHandle<TOutputHandle>>
    {
        using BaseType = BaseEvalItem<DeviceTypeFromHandle<TOutputHandle>>;
    public:
        template <typename TAuxParams>
        EvalItem(TWeightHandle weightHandle, TInputHandle inputHandle,
                 TOutputHandle outputHandle, const TAuxParams&)
            : BaseType(std::type_index(typeid(EvalItem)),
                       {weightHandle.DataPtr(), inputHandle.DataPtr()},
                       outputHandle.DataPtr())
            , m_weightHandle(std::move(weightHandle))
            , m_inputHandle(std::move(inputHandle))
            , m_outputHandle(std::move(outputHandle))
        {}
        
        const TWeightHandle m_weightHandle;
        const TInputHandle m_inputHandle;
        TOutputHandle m_outputHandle;
    };
    
    template <typename TWeightHandle, typename TInputHandle, typename TOutputHandle>
    class EvalGroup : public TrivalEvalGroup<EvalItem<TWeightHandle, TInputHandle, TOutputHandle>>
    {
        using EvalItemType = EvalItem<TWeightHandle, TInputHandle, TOutputHandle>;
    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            const auto& weight = evalItem.m_weightHandle.Data();
            const auto& in = evalItem.m_inputHandle.Data();
            assert(weight.Shape() == in.Shape());

            using ResType = typename TOutputHandle::DataType;
            using ElementType = typename ResType::ElementType;
            ResType out;

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
            evalItem.m_outputHandle.SetData(std::move(out));
        }
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
    using type = OperCalAlgoChain<TailCalculator<OperNLLLoss::NSCaseGen::EvalItem, OperNLLLoss::NSCaseGen::EvalGroup>>;
};

template <typename TWeight, typename TInput,
          typename = std::enable_if_t<IsValidOper<OpTags::NLLLoss, TWeight, TInput>>>
auto NLLLoss(TWeight&& p_weight, TInput&& p_input)
{
    using ResType = Operator<OpTags::NLLLoss, RemConstRef<TWeight>, RemConstRef<TInput>>;
    return ResType(std::forward<TWeight>(p_weight), std::forward<TInput>(p_input));
}
}

namespace MetaNN
{
namespace OperNLLLossGrad::NSCaseGen
{
    template <typename TGradHandle, typename TWeightHandle, typename TInputHandle, typename TOutputHandle>
    class EvalItem : public BaseEvalItem<DeviceTypeFromHandle<TOutputHandle>>
    {
        using BaseType = BaseEvalItem<DeviceTypeFromHandle<TOutputHandle>>;
    public:
        template <typename TAuxParams>
        EvalItem(TGradHandle gradHandle, TWeightHandle weightHandle, TInputHandle inputHandle,
                 TOutputHandle outputHandle, const TAuxParams&)
            : BaseType(std::type_index(typeid(EvalItem)),
                       {gradHandle.DataPtr(), weightHandle.DataPtr(), inputHandle.DataPtr()},
                       outputHandle.DataPtr())
            , m_gradHandle(std::move(gradHandle))
            , m_weightHandle(std::move(weightHandle))
            , m_inputHandle(std::move(inputHandle))
            , m_outputHandle(std::move(outputHandle))
        {}
        
        const TGradHandle m_gradHandle;
        const TWeightHandle m_weightHandle;
        const TInputHandle m_inputHandle;
        TOutputHandle m_outputHandle;
    };

    template <typename TGradHandle, typename TWeightHandle, typename TInputHandle, typename TOutputHandle>
    class EvalGroup : public TrivalEvalGroup<EvalItem<TGradHandle, TWeightHandle, TInputHandle, TOutputHandle>>
    {
        using EvalItemType = EvalItem<TGradHandle, TWeightHandle, TInputHandle, TOutputHandle>;
    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            const auto& grad = evalItem.m_gradHandle.Data();
            const auto& weight = evalItem.m_weightHandle.Data();
            const auto& in = evalItem.m_inputHandle.Data();
            assert(weight.Shape() == in.Shape());

            using ResType = typename TOutputHandle::DataType;
            using ElementType = typename ResType::ElementType;
            ResType out(weight.Shape());

            const size_t count = in.Shape().Count();
            const size_t cardinalCount = in.Shape().CardinalShape().Count();
            assert(count % cardinalCount == 0);
            const size_t loopCount = count / cardinalCount;

            auto low_grad = LowerAccess(grad);
            const ElementType* mem_grad = low_grad.RawMemory();
            auto low_in = LowerAccess(in);
            const ElementType* mem_in = low_in.RawMemory();
            auto low_weight = LowerAccess(weight);
            const ElementType* mem_weight = low_weight.RawMemory();
            auto low_out = LowerAccess(out);
            ElementType* mem_out = low_out.MutableRawMemory();

            static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");

            for (size_t loop = 0; loop < loopCount; ++loop)
            {
                ElementType ngv = -mem_grad[loop];
                for (size_t i = 0; i < cardinalCount; ++i)
                {
                    mem_out[i] = ngv * mem_weight[i] / mem_in[i];
                }
                mem_out += cardinalCount;
                mem_weight += cardinalCount;
                mem_in += cardinalCount;
            }

            evalItem.m_outputHandle.SetData(std::move(out));
        }
    };
}

template <typename TGrad, typename TWeight, typename TInput>
struct OperCategory_<OpTags::NLLLossGrad, TGrad, TWeight, TInput>
    : public PickCommonCategory_<TWeight, TInput>
{};

template <typename TCate>
class OperShapeInfo<OpTags::NLLLossGrad, TCate>
    : public GenLossBPOperShapeInfo<TCate>
{
    using TBase = GenLossBPOperShapeInfo<TCate>;
public:
    using TBase::TBase;
};

template <>
struct OperSeq_<OpTags::NLLLossGrad>
{
    using type = OperCalAlgoChain<TailCalculator<OperNLLLossGrad::NSCaseGen::EvalItem, OperNLLLossGrad::NSCaseGen::EvalGroup>>;
};

// interface
template <typename TGrad, typename TWeight, typename TInput,
          typename = std::enable_if_t<IsValidLossBP<OpTags::NLLLossGrad, TGrad, TWeight, TInput>>>
auto NLLLossGrad(TGrad&& p_grad, TWeight&& p_weight, TInput&& p_input)
{
    using ResType = Operator<OpTags::NLLLossGrad, RemConstRef<TGrad>, RemConstRef<TWeight>, RemConstRef<TInput>>;
    return ResType(std::forward<TGrad>(p_grad), std::forward<TWeight>(p_weight), std::forward<TInput>(p_input));
}
}