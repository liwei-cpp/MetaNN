#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <MetaNN/operation/facilities/_.h>
#include <MetaNN/policies/_.h>
#include <MetaNN/facilities/_.h>
#include <cassert>
#include <cmath>
#include <type_traits>

namespace MetaNN::OpTags
{
    struct Softmax;
    struct SoftmaxGrad;

    // optimization assistant tags
    struct NLLLossGrad;
}

namespace MetaNN
{
namespace OperSoftmax::NSCaseGen
{
    template <typename TInputHandle, typename TOutputHandle, typename TPolicy>
    class EvalItem : public BaseEvalItem
    {
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

    template <typename TInputHandle, typename TOutputHandle, typename TPolicy>
    class EvalGroup : public TrivialEvalGroup<EvalItem<TInputHandle, TOutputHandle, TPolicy>>
    {
        using EvalItemType = EvalItem<TInputHandle, TOutputHandle, TPolicy>;
        using ResType = typename TOutputHandle::DataType;
        using ElementType = typename ResType::ElementType;

    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            const auto& in = evalItem.m_inputHandle.Data();
            ResType out(in.Shape());
            
            constexpr size_t modDimNum = PolicySelect<DimPolicy, TPolicy>::ModifyDimNum;
            size_t loopCount = 1;
            for (size_t i = 0;
                 i < CategoryTagFromHandle<TOutputHandle>::DimNum - modDimNum;
                 ++i)
            {
                loopCount *= in.Shape()[i];
            }
            assert(in.Shape().Count() % loopCount == 0);
            const size_t modCount = in.Shape().Count() / loopCount;

            auto low_in = LowerAccess(in);
            const ElementType* mem_in = low_in.RawMemory();

            auto low_out = LowerAccess(out);
            ElementType* mem_out = low_out.MutableRawMemory();

            static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");

            for (size_t i = 0; i < loopCount; ++i)
            {
                EvalSoftmax(mem_out, mem_in, modCount);
                mem_out += modCount;
                mem_in += modCount;
            }
            evalItem.m_outputHandle.SetData(std::move(out));
        }
    private:
        void EvalSoftmax(ElementType* out, const ElementType* in, const size_t len)
        {
            ElementType maxElem = *std::max_element(in, in + len);
            ElementType sum{};

            for (size_t i = 0; i < len; ++i)
            {
                out[i] = exp(in[i] - maxElem);
                sum += out[i];
            }

            for (size_t i = 0; i < len; ++i)
            {
                out[i] /= sum;
            }
        }
    };
}

    template <>
    struct OperSeq_<OpTags::Softmax>
    {
        using type = OperCalAlgoChain<TailCalculator<OperSoftmax::NSCaseGen::EvalItem,
                                                     OperSoftmax::NSCaseGen::EvalGroup,
                                                     PolicyContainer<PPassPolicy>>>;
    };

    template <typename TPolicy = PolicyContainer<>,
              typename TP,
              std::enable_if_t<IsValidOper<OpTags::Softmax, TP>>* = nullptr>
    auto Softmax(TP&& p_m)
    {
        constexpr size_t modDimNum = PolicySelect<DimPolicy, TPolicy>::ModifyDimNum;
        static_assert(DataCategory<TP>::DimNum >= modDimNum);
        using rawM = RemConstRef<TP>;
        using ResType = Operation<OpTags::Softmax, OperandContainer<rawM>,
                                  PolicyContainer<PModifyDimNumIs<modDimNum>>>;
        return ResType(std::forward<TP>(p_m));
    }
}

namespace MetaNN
{
namespace OperSoftmaxGrad::NSCaseGen
{
    template <typename TGradHandle, typename TInputHandle, typename TOutputHandle, typename TPolicy>
    class EvalItem : public BaseEvalItem
    {
    public:
        EvalItem(TGradHandle gradHandle, TInputHandle oriHandle, TOutputHandle outputHandle)
            : BaseEvalItem(TypeID<EvalItem>(),
                           {gradHandle.DataPtr(), oriHandle.DataPtr()}, outputHandle.DataPtr())
            , m_gradHandle(std::move(gradHandle))
            , m_inputHandle(std::move(oriHandle))
            , m_outputHandle(std::move(outputHandle))
        {}
        
        const TGradHandle m_gradHandle;
        const TInputHandle m_inputHandle;
        TOutputHandle m_outputHandle;
    };

    template <typename TGradHandle, typename TInputHandle, typename TOutputHandle, typename TPolicy>
    class EvalGroup : public TrivialEvalGroup<EvalItem<TGradHandle, TInputHandle, TOutputHandle, TPolicy>>
    {
        using EvalItemType = EvalItem<TGradHandle, TInputHandle, TOutputHandle, TPolicy>;
        using ResType = typename TOutputHandle::DataType;
        using ElementType = typename ResType::ElementType;

    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            const auto& grad = evalItem.m_gradHandle.Data();
            const auto& in = evalItem.m_inputHandle.Data();
            assert(in.Shape() == grad.Shape());

            ResType out(in.Shape());
            constexpr size_t modDimNum = PolicySelect<DimPolicy, TPolicy>::ModifyDimNum;
            size_t loopCount = 1;
            for (size_t i = 0;
                 i < CategoryTagFromHandle<TOutputHandle>::DimNum - modDimNum;
                 ++i)
            {
                loopCount *= in.Shape()[i];
            }
            assert(in.Shape().Count() % loopCount == 0);
            const size_t modCount = in.Shape().Count() / loopCount;

            auto low_grad = LowerAccess(grad);
            const ElementType* mem_grad = low_grad.RawMemory();
            auto low_in = LowerAccess(in);
            const ElementType* mem_in = low_in.RawMemory();

            auto low_out = LowerAccess(out);
            ElementType* mem_out = low_out.MutableRawMemory();

            static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");

            for (size_t i = 0; i < loopCount; ++i)
            {
                EvalSingleLoop(mem_out, mem_grad, mem_in, modCount);
                mem_out += modCount;
                mem_in += modCount;
                mem_grad += modCount;
            }
            evalItem.m_outputHandle.SetData(std::move(out));
        }
    private:
        void EvalSingleLoop(ElementType* out, const ElementType* grad, const ElementType* in, const size_t len)
        {
            ElementType sum{};
            for (size_t i = 0; i < len; ++i)
            {
                sum += grad[i] * in[i];
            }
            for (size_t i = 0; i < len; ++i)
            {
                out[i] = in[i] * (grad[i] - sum);
            }
        }
    };
}

namespace OperSoftmaxGrad::NSCaseNLLLossGrad
{
    template <typename T1, typename T2>
    struct Valid_
    {
        constexpr static bool value = false;
    };
    
    template <typename TLossOperands, typename TSoftmaxOperand>
    struct Valid_<Operation<OpTags::NLLLossGrad, TLossOperands, PolicyContainer<>>,
                  TSoftmaxOperand>
    {
        using TCheckOperand = Sequential::At<TLossOperands, 2>;
        constexpr static bool value = std::is_same_v<TCheckOperand, TSoftmaxOperand>;
    };
    
    template <typename TGradHandle, typename TWeightHandle, 
              typename TSoftmaxHandle, typename TOutputHandle, typename TPolicy>
    class EvalItem : public BaseEvalItem
    {
    public:
        EvalItem(TGradHandle gradHandle, TWeightHandle weightHandle,
                    TSoftmaxHandle softmaxHandle, TOutputHandle outputHandle)
            : BaseEvalItem(TypeID<EvalItem>(),
                           {gradHandle.DataPtr(), weightHandle.DataPtr(), softmaxHandle.DataPtr()},
                           outputHandle.DataPtr())
            , m_gradHandle(std::move(gradHandle))
            , m_weightHandle(std::move(weightHandle))
            , m_softmaxHandle(std::move(softmaxHandle))
            , m_outputHandle(std::move(outputHandle))
        {}
        
        TGradHandle    m_gradHandle;
        TWeightHandle  m_weightHandle;
        TSoftmaxHandle m_softmaxHandle;
        TOutputHandle  m_outputHandle;
    };
    
    template <typename TGradHandle, typename TWeightHandle, 
              typename TSoftmaxHandle, typename TOutputHandle, typename TPolicy>
    class EvalGroup : public TrivialEvalGroup<EvalItem<TGradHandle, TWeightHandle, TSoftmaxHandle, TOutputHandle, TPolicy>>
    {
        using EvalItemType = EvalItem<TGradHandle, TWeightHandle, TSoftmaxHandle, TOutputHandle, TPolicy>;
    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            using ResType = typename TOutputHandle::DataType;
            using ElementType = typename ResType::ElementType;
            
            const auto& grad = evalItem.m_gradHandle.Data();
            const auto& weight = evalItem.m_weightHandle.Data();
            const auto& softmaxRes = evalItem.m_softmaxHandle.Data();
            assert(weight.Shape() == softmaxRes.Shape());

            constexpr size_t modDimNum = PolicySelect<DimPolicy, TPolicy>::ModifyDimNum;
            size_t loopCount = 1;
            for (size_t i = 0;
                 i < CategoryTagFromHandle<TOutputHandle>::DimNum - modDimNum;
                 ++i)
            {
                loopCount *= softmaxRes.Shape()[i];
            }
            assert(softmaxRes.Shape().Count() % loopCount == 0);
            const size_t modCount = softmaxRes.Shape().Count() / loopCount;
            
            ResType out(weight.Shape());

            const ElementType gradValue = grad.Value();
            auto low_weight = LowerAccess(weight);
            const ElementType* mem_weight = low_weight.RawMemory();
            auto low_softmaxRes = LowerAccess(softmaxRes);
            const ElementType* mem_softmaxRes = low_softmaxRes.RawMemory();

            auto low_out = LowerAccess(out);
            ElementType* mem_out = low_out.MutableRawMemory();

            for (size_t curLoop = 0; curLoop < loopCount; ++curLoop)
            {
                ElementType sum{};
                for (size_t i = 0; i < modCount; ++i)
                {
                    sum += mem_weight[i];
                }

                for (size_t i = 0; i < modCount; ++i)
                {
                    mem_out[i] = (mem_softmaxRes[i] * sum - mem_weight[i]) * gradValue;
                }
                mem_out += modCount;
                mem_weight += modCount;
                mem_softmaxRes += modCount;
            }
            evalItem.m_outputHandle.SetData(std::move(out));
        }
    };

    
    struct Calculator
    {
        template <typename TCaseTail, typename TEvalRes, typename TOp>
        static void EvalRegister(TEvalRes& evalRes, const TOp& oper)
        {
            if constexpr (!Valid_<typename TOp::template OperandType<0>,
                                  typename TOp::template OperandType<1>>::value)
            {
                using THead = Sequential::Head<TCaseTail>;
                using TTail = Sequential::Tail<TCaseTail>;
                THead::template EvalRegister<TTail>(evalRes, oper);
            }
            else
            {
                const auto& oper0 = oper.template Operand<0>();
                const auto& oper1 = oper.template Operand<1>();
                const auto& softmax_res = oper0.template Operand<2>();
                if (softmax_res != oper1)
                {
                    using THead = Sequential::Head<TCaseTail>;
                    using TTail = Sequential::Tail<TCaseTail>;
                    THead::template EvalRegister<TTail>(evalRes, oper);
                    return;
                }

                auto outHandle = evalRes.Handle();

                auto gradHandle = oper0.template Operand<0>().EvalRegister();
                auto softmaxHandle = oper1.EvalRegister();
                const auto operWeight = oper0.template Operand<1>();

                auto weightHandle = operWeight.EvalRegister();
                using TItem = EvalItem<RemConstRef<decltype(gradHandle)>,
                                       RemConstRef<decltype(weightHandle)>,
                                       RemConstRef<decltype(softmaxHandle)>,
                                       RemConstRef<decltype(outHandle)>,
                                       typename TOp::Policies>;
                using TGroup = EvalGroup<RemConstRef<decltype(gradHandle)>,
                                         RemConstRef<decltype(weightHandle)>,
                                         RemConstRef<decltype(softmaxHandle)>,
                                         RemConstRef<decltype(outHandle)>,
                                         typename TOp::Policies>;
                using EvalDispatcher = TrivialEvalItemDispatcher<TGroup>;
                
                auto item = std::make_unique<TItem>(std::move(gradHandle), std::move(weightHandle),
                                                    std::move(softmaxHandle), std::move(outHandle));
                EvalPlan::Inst().Register<EvalDispatcher>(std::move(item));
        }
    }
};
}

    template <>
    struct OperSeq_<OpTags::SoftmaxGrad>
    {
        using type = OperCalAlgoChain<OperSoftmaxGrad::NSCaseNLLLossGrad::Calculator,
                                      TailCalculator<OperSoftmaxGrad::NSCaseGen::EvalItem,
                                                     OperSoftmaxGrad::NSCaseGen::EvalGroup,
                                                     PolicyContainer<PPassPolicy>>>;
    };

    template <typename TPolicy = PolicyContainer<>, typename TGrad, typename TInput,
              std::enable_if_t<IsValidOper<OpTags::SoftmaxGrad, TGrad, TInput>>* = nullptr>
    auto SoftmaxGrad(TGrad&& p_grad, TInput&& p_input)
    {
        static_assert(std::is_same_v<DataCategory<TGrad>, DataCategory<TInput>>);
        constexpr size_t modDimNum = PolicySelect<DimPolicy, TPolicy>::ModifyDimNum;
        static_assert(DataCategory<TInput>::DimNum >= modDimNum);
        
        using rawOp1 = RemConstRef<TGrad>;
        using rawOp2 = RemConstRef<TInput>;
        using ResType = Operation<OpTags::SoftmaxGrad, OperandContainer<rawOp1, rawOp2>,
                                  PolicyContainer<PModifyDimNumIs<modDimNum>>>;
        return ResType(std::forward<TGrad>(p_grad), std::forward<TInput>(p_input));
    }
}
