#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <MetaNN/operators/facilities/tail_calculator.h>
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
template <typename TInputHandle, typename TOutputHandle>
class EvalUnit : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
    using ResType = typename TOutputHandle::DataType;
    using ElementType = typename ResType::ElementType;

public:
    template <typename TAuxParams>
    EvalUnit(TInputHandle oriHandle, TOutputHandle outputHandle, const TAuxParams&)
        : m_inputHandle(std::move(oriHandle))
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        const auto& in = m_inputHandle.Data();
        ResType out(in.Shape());
        
        const size_t count = in.Shape().Count();
        assert(count == out.Shape().Count());
        const size_t matrixSize = in.Shape().RowNum() * in.Shape().ColNum();
        assert(count % matrixSize == 0);
        const size_t loopCount = count / matrixSize;
        
        auto low_in = LowerAccess(in);
        ElementType* mem_in = low_in.MutableRawMemory();

        auto low_out = LowerAccess(out);
        ElementType* mem_out = low_out.MutableRawMemory();
                
        static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
        
        for (size_t i = 0; i < loopCount; ++i)
        {
            EvalMatrix(mem_out, mem_in, matrixSize);
            mem_out += matrixSize;
            mem_in += matrixSize;
        }
        m_outputHandle.SetData(std::move(out));
    }
    
private:
    void EvalMatrix(ElementType* out, ElementType* in, const size_t len)
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
private:
    const TInputHandle m_inputHandle;
    TOutputHandle m_outputHandle;
};
}

template <typename TOperand>
constexpr bool IsValidOper<OpTags::Softmax, TOperand> =
    IsMatrix<TOperand> ||
    IsBatchMatrix<TOperand>;

template <>
struct OperSeq_<OpTags::Softmax>
{
    using type = OperSeqContainer<TailCalculator<OperSoftmax::NSCaseGen::EvalUnit>>;
};

template <typename TP,
          typename = std::enable_if_t<IsValidOper<OpTags::Softmax, TP>>>
auto Softmax(TP&& p_m)
{
    using rawM = RemConstRef<TP>;
    using ResType = Operator<OpTags::Softmax, rawM>;
    return ResType(std::forward<TP>(p_m));
}
}

namespace MetaNN
{
namespace OperSoftmaxGrad::NSCaseNLLLossGrad
{
template <typename T1, typename T2>
constexpr bool Valid = false;

template <typename T1, typename T2, typename T3>
constexpr bool Valid<Operator<OpTags::NLLLossGrad, T1, T2, T3>,
                     T3> = true;

template <typename TGradHandle, typename TSoftmaxHandle, typename TOutputHandle>
class EuOneHotWeight : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
    using ResType = typename TOutputHandle::DataType;
    using ElementType = typename ResType::ElementType;
public:
    EuOneHotWeight(TGradHandle gradHandle, TSoftmaxHandle softmaxHandle,
                   TOutputHandle outputHandle, size_t hotPos)
        : m_gradHandle(std::move(gradHandle))
        , m_softmaxHandle(std::move(softmaxHandle))
        , m_outputHandle(std::move(outputHandle))
        , m_hotPos(hotPos)
    {}
        
    void Eval() override
    {
        const auto& grad = m_gradHandle.Data().Value();
        const auto& softmaxRes = m_softmaxHandle.Data();
        assert(softmaxRes.Shape().RowNum() == 1);
        
        size_t colNum = softmaxRes.Shape().ColNum();
        assert(m_hotPos < colNum);
        
        ResType res(1, colNum);

        auto mem_res = LowerAccess(res);
        ElementType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < colNum; ++i)
        {
            r[i] = softmaxRes(0, i) * grad;
        }
        r[m_hotPos] -= grad;
        m_outputHandle.SetData(std::move(res));
    }

private:
    TGradHandle    m_gradHandle;
    TSoftmaxHandle m_softmaxHandle;
    TOutputHandle  m_outputHandle;
    size_t m_hotPos;
};

template <typename TGradHandle, typename TWeightHandle, 
          typename TSoftmaxHandle, typename TOutputHandle>
class EuGenWeight : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
    using ResType = typename TOutputHandle::DataType;
    using ElementType = typename ResType::ElementType;
public:
    EuGenWeight(TGradHandle gradHandle, TWeightHandle weightHandle,
                TSoftmaxHandle softmaxHandle, TOutputHandle outputHandle)
        : m_gradHandle(std::move(gradHandle))
        , m_weightHandle(std::move(weightHandle))
        , m_softmaxHandle(std::move(softmaxHandle))
        , m_outputHandle(std::move(outputHandle))
    {}

    void Eval() override
    {
        const auto& grad = m_gradHandle.Data();
        const size_t loopCount = grad.Shape().Count();
        
        const auto& weight = m_weightHandle.Data();
        const auto& softmaxRes = m_softmaxHandle.Data();
        assert(weight.Shape() == softmaxRes.Shape());
        
        const size_t cardinalCount = weight.Shape().CardinalShape().Count();
        assert(weight.Shape().Count() % cardinalCount == 0);
        assert(loopCount == weight.Shape().Count() / cardinalCount);
        
        ResType out(weight.Shape());

        auto low_grad = LowerAccess(grad);
        ElementType* mem_grad = low_grad.RawMemory();
        auto low_weight = LowerAccess(weight);
        ElementType* mem_weight = low_weight.RawMemory();
        auto low_softmaxRes = LowerAccess(softmaxRes);
        ElementType* mem_softmaxRes = low_softmaxRes.RawMemory();
        
        auto low_out = LowerAccess(out);
        ElementType* mem_out = low_out.RawMemory();
        
        for (size_t curLoop = 0; curLoop < loopCount; ++curLoop)
        {
            ElementType sum{};
            ElementType curGrad = *mem_grad;
            for (size_t i = 0; i < cardinalCount; ++i)
            {
                sum += mem_weight[i];
            }

            for (size_t i = 0; i < cardinalCount; ++i)
            {
                mem_out[i] = (mem_softmaxRes[i] * sum - mem_weight[i]) * curGrad;
            }
            mem_out += cardinalCount;
            mem_weight += cardinalCount;
            mem_softmaxRes += cardinalCount;
            ++mem_grad;
        }
        m_outputHandle.SetData(std::move(out));
    }
private:
    TGradHandle    m_gradHandle;
    TWeightHandle  m_weightHandle;
    TSoftmaxHandle m_softmaxHandle;
    TOutputHandle  m_outputHandle;
};

struct Calculator
{
    template <typename TCaseTail, typename TEvalRes, typename TOp>
    static void EvalRegister(TEvalRes& evalRes, const TOp& oper)
    {
        using TOperand0 = RemConstRef<decltype(oper.template Operand<0>())>;
        using TOperand1 = RemConstRef<decltype(oper.template Operand<1>())>;
        if constexpr (!Valid<TOperand0, TOperand1>)
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
            using TElement = ElementTypeFromHandle<decltype(outHandle)>;
            using TDevice = DeviceTypeFromHandle<decltype(outHandle)>;

            auto gradHandle = oper0.template Operand<0>().EvalRegister();
            auto softmaxHandle = oper1.EvalRegister();
            const auto operWeight = oper0.template Operand<1>();
            if constexpr (std::is_same_v<RemConstRef<decltype(operWeight)>, OneHotVector<TElement, TDevice>>)
            {
                using EvalUnit = EuOneHotWeight<RemConstRef<decltype(gradHandle)>,
                                                RemConstRef<decltype(softmaxHandle)>,
                                                RemConstRef<decltype(outHandle)>>;
                using GroupType = TrivalEvalGroup<EvalUnit>;

                auto dataPtr = outHandle.DataPtr();
                auto depVec = {gradHandle.DataPtr(), softmaxHandle.DataPtr()};
                EvalUnit unit(std::move(gradHandle), std::move(softmaxHandle), std::move(outHandle), operWeight.HotPos());
                EvalPlan<TDevice>::template Register<GroupType>(std::move(unit), dataPtr, std::move(depVec));
            }
            // TODO: add specialization for batch one hot vector and one hot vector sequence
            else
            {
                auto weightHandle = operWeight.EvalRegister();
                using EvalUnit = EuGenWeight<RemConstRef<decltype(gradHandle)>,
                                             RemConstRef<decltype(weightHandle)>,
                                             RemConstRef<decltype(softmaxHandle)>,
                                             RemConstRef<decltype(outHandle)>>;
                using GroupType = TrivalEvalGroup<EvalUnit>;

                auto dataPtr = outHandle.DataPtr();
                auto depVec = {gradHandle.DataPtr(), weightHandle.DataPtr(), softmaxHandle.DataPtr()};
                EvalUnit unit(std::move(gradHandle), std::move(weightHandle),
                              std::move(softmaxHandle), std::move(outHandle));
                EvalPlan<TDevice>::template Register<GroupType>(std::move(unit), dataPtr, std::move(depVec));
            }
        }
    }
};
}

namespace OperSoftmaxGrad::NSCaseGen
{
template <typename TGradHandle, typename TInputHandle, typename TOutputHandle>
class EvalUnit : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
    using ResType = typename TOutputHandle::DataType;
    using ElementType = typename ResType::ElementType;
public:
    template <typename TAuxParams>
    EvalUnit(TGradHandle gradHandle, TInputHandle inputHandle, TOutputHandle outputHandle, const TAuxParams&)
        : m_gradHandle(std::move(gradHandle))
        , m_inputHandle(std::move(inputHandle))
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        const auto& grad = m_gradHandle.Data();
        const auto& in = m_inputHandle.Data();
        assert(grad.Shape() == in.Shape());
        
        ResType out(in.Shape());

        const size_t count = in.Shape().Count();
        assert(count == out.Shape().Count());
        const size_t matrixSize = in.Shape().RowNum() * in.Shape().ColNum();
        assert(count % matrixSize == 0);
        const size_t loopCount = count / matrixSize;
        
        auto low_grad = LowerAccess(grad);
        ElementType* mem_grad = low_grad.MutableRawMemory();
        auto low_in = LowerAccess(in);
        ElementType* mem_in = low_in.MutableRawMemory();

        auto low_out = LowerAccess(out);
        ElementType* mem_out = low_out.MutableRawMemory();
                
        static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
        
        for (size_t i = 0; i < loopCount; ++i)
        {
            EvalSingleLoop(mem_out, mem_grad, mem_in, matrixSize);
            mem_out += matrixSize;
            mem_in += matrixSize;
            mem_grad += matrixSize;
        }
        m_outputHandle.SetData(std::move(out));
    }

private:
    void EvalSingleLoop(ElementType* out, ElementType* grad, ElementType* in, const size_t len)
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
private:
    const TGradHandle m_gradHandle;
    const TInputHandle m_inputHandle;
    TOutputHandle m_outputHandle;
};
}

template <>
struct OperSeq_<OpTags::SoftmaxGrad>
{
    using type = OperSeqContainer<OperSoftmaxGrad::NSCaseNLLLossGrad::Calculator,
                                  TailCalculator<OperSoftmaxGrad::NSCaseGen::EvalUnit>>;
};

template <typename TOperandGrad, typename TOperandInput>
constexpr bool IsValidOper<OpTags::SoftmaxGrad, TOperandGrad, TOperandInput> =
    (IsMatrix<TOperandGrad> && IsMatrix<TOperandInput>) ||
    (IsBatchMatrix<TOperandGrad> && IsBatchMatrix<TOperandInput>);

template <typename TGrad, typename TInput,
          typename = std::enable_if_t<IsValidOper<OpTags::SoftmaxGrad, TGrad, TInput>>>
auto SoftmaxGrad(TGrad&& p_grad, TInput&& p_input)
{
    if (p_grad.Shape() != p_input.Shape())
    {
        throw std::runtime_error("SoftmaxGrad error: operands' shape mismatch.");
    }
    
    using rawOp1 = RemConstRef<TGrad>;
    using rawOp2 = RemConstRef<TInput>;
    using ResType = Operator<OpTags::SoftmaxGrad, rawOp1, rawOp2>;
    return ResType(std::forward<TGrad>(p_grad), std::forward<TInput>(p_input));
}
}