#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <MetaNN/facilities/_.h>
#include <MetaNN/operation/facilities/_.h>
#include <MetaNN/operation/nn/facilities/organizer.h>
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
    template <typename TTruthHandle, typename TPredHandle, typename TOutputHandle>
    class EvalItem : public BaseEvalItem
    {
    public:
        using CategoryTag = CategoryTagFromHandle<TOutputHandle>;

        EvalItem(TTruthHandle truthHandle, TPredHandle predHandle, TOutputHandle outputHandle)
            : BaseEvalItem(TypeID<EvalItem>(),
                           {truthHandle.DataPtr(), predHandle.DataPtr()},
                           outputHandle.DataPtr())
            , m_truthHandle(std::move(truthHandle))
            , m_predHandle(std::move(predHandle))
            , m_outputHandle(std::move(outputHandle))
        {}
        
        const TTruthHandle m_truthHandle;
        const TPredHandle m_predHandle;
        TOutputHandle m_outputHandle;
    };
    
    template <typename TTruthHandle, typename TPredHandle, typename TOutputHandle>
    class EvalGroup : public TrivialEvalGroup<EvalItem<TTruthHandle, TPredHandle, TOutputHandle>>
    {
        using EvalItemType = EvalItem<TTruthHandle, TPredHandle, TOutputHandle>;
    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            const auto& truth = evalItem.m_truthHandle.Data();
            const auto& pred = evalItem.m_predHandle.Data();
            
            const size_t count = truth.Shape().Count();
            assert(count == pred.Shape().Count());

            using ResType = typename TOutputHandle::DataType;
            using ElementType = typename ResType::ElementType;
            ResType out;

            auto low_pred = LowerAccess(pred);
            ElementType* mem_pred = low_pred.MutableRawMemory();

            auto low_truth = LowerAccess(truth);
            ElementType* mem_truth = low_truth.MutableRawMemory();

            static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");

            ElementType res{};
            ElementType sum_truth{};
            for (size_t i = 0; i < count; ++i)
            {
                res -= mem_truth[i] * log(mem_pred[i]);
                sum_truth += mem_truth[i];
            }

            out.SetValue(res / sum_truth);
            evalItem.m_outputHandle.SetData(std::move(out));
        }
    };
}

template <typename TPolicies, typename TTruth, typename TPred>
struct OperCategory_<OpTags::NLLLoss, TPolicies, TTruth, TPred>
    : public GenLossOperCategory_
{};

template <typename TPolicies>
class OperShapeInfo<OpTags::NLLLoss, CategoryTags::Tensor<0>, TPolicies>
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

template <typename TTruth, typename TPred,
          std::enable_if_t<IsValidOper<OpTags::NLLLoss, TTruth, TPred>>* = nullptr>
auto NLLLoss(TTruth&& p_truth, TPred&& p_pred)
{
    static_assert(std::is_same_v<DataCategory<TTruth>, DataCategory<TPred>>);
    using ResType = Operation<OpTags::NLLLoss, OperandContainer<RemConstRef<TTruth>, RemConstRef<TPred>>>;
    return ResType(std::forward<TTruth>(p_truth), std::forward<TPred>(p_pred));
}
}

namespace MetaNN
{
namespace OperNLLLossGrad::NSCaseGen
{
    template <typename TGradHandle, typename TTruthHandle, typename TPredHandle, typename TOutputHandle>
    class EvalItem : public BaseEvalItem
    {
    public:
        using CategoryTag = CategoryTagFromHandle<TOutputHandle>;

        EvalItem(TGradHandle gradHandle, TTruthHandle truthHandle, TPredHandle predHandle,
                 TOutputHandle outputHandle)
            : BaseEvalItem(TypeID<EvalItem>(),
                           {gradHandle.DataPtr(), truthHandle.DataPtr(), predHandle.DataPtr()},
                           outputHandle.DataPtr())
            , m_gradHandle(std::move(gradHandle))
            , m_truthHandle(std::move(truthHandle))
            , m_predHandle(std::move(predHandle))
            , m_outputHandle(std::move(outputHandle))
        {}
        
        const TGradHandle m_gradHandle;
        const TTruthHandle m_truthHandle;
        const TPredHandle m_predHandle;
        TOutputHandle m_outputHandle;
    };

    template <typename TGradHandle, typename TTruthHandle, typename TPredHandle, typename TOutputHandle>
    class EvalGroup : public TrivialEvalGroup<EvalItem<TGradHandle, TTruthHandle, TPredHandle, TOutputHandle>>
    {
        using EvalItemType = EvalItem<TGradHandle, TTruthHandle, TPredHandle, TOutputHandle>;
    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            const auto& grad = evalItem.m_gradHandle.Data();
            const auto& truth = evalItem.m_truthHandle.Data();
            const auto& pred = evalItem.m_predHandle.Data();
            assert(truth.Shape() == pred.Shape());

            using ResType = typename TOutputHandle::DataType;
            using ElementType = typename ResType::ElementType;
            ResType out(truth.Shape());

            const size_t count = truth.Shape().Count();

            const ElementType neg_grad = -grad.Value();
            auto low_pred = LowerAccess(pred);
            const ElementType* mem_pred = low_pred.RawMemory();
            auto low_truth = LowerAccess(truth);
            const ElementType* mem_truth = low_truth.RawMemory();
            auto low_out = LowerAccess(out);
            ElementType* mem_out = low_out.MutableRawMemory();

            static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");

            ElementType sum_truth{};
            for (size_t i = 0; i < count; ++i)
            {
                sum_truth += mem_truth[i];
            }

            const ElementType scal = neg_grad / sum_truth;
            for (size_t i = 0; i < count; ++i)
            {
                mem_out[i] = scal * mem_truth[i] / mem_pred[i];
            }

            evalItem.m_outputHandle.SetData(std::move(out));
        }
    };
}

template <typename TPolicies, typename TGrad, typename TWeight, typename TInput>
struct OperCategory_<OpTags::NLLLossGrad, TPolicies, TGrad, TWeight, TInput>
    : public PickCommonCategory_<TWeight, TInput>
{};

template <>
struct OperSeq_<OpTags::NLLLossGrad>
{
    using type = OperCalAlgoChain<TailCalculator<OperNLLLossGrad::NSCaseGen::EvalItem, OperNLLLossGrad::NSCaseGen::EvalGroup>>;
};

// interface
template <typename TGrad, typename TTruth, typename TPred,
          std::enable_if_t<IsValidOper<OpTags::NLLLossGrad, TGrad, TTruth, TPred>>* = nullptr>
auto NLLLossGrad(TGrad&& p_grad, TTruth&& p_truth, TPred&& p_pred)
{
    static_assert(IsScalar<TGrad>);
    static_assert(std::is_same_v<DataCategory<TPred>, DataCategory<TTruth>>);

    using ResType = Operation<OpTags::NLLLossGrad, OperandContainer<RemConstRef<TGrad>, RemConstRef<TTruth>, RemConstRef<TPred>>>;
    return ResType(std::forward<TGrad>(p_grad), std::forward<TTruth>(p_truth), std::forward<TPred>(p_pred));
}
}