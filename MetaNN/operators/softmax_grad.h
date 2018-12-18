#pragma once
/*
#include <MetaNN/operators/operators.h>
#include <stdexcept>

namespace MetaNN
{
namespace NSVecSoftmaxDerivative
{
namespace CaseNLL
{
template <typename T1, typename T2>
constexpr bool Valid = false;

template <typename T1, typename T2, typename T3>
constexpr bool Valid<TernaryOp<TernaryOpTags::NegativeLogLikelihoodDerivative,
                               T1, T2, T3>,
                     T3> = true;
                     
template <typename TOperHandle1, typename TOperHandle2, typename TElem, typename TDevice>
class Case1EvalUnit;

template <typename TOperHandle1, typename TOperHandle2, typename TElem>
class Case1EvalUnit<TOperHandle1, TOperHandle2, TElem, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;

    Case1EvalUnit(TOperHandle1 grad,
                  size_t hotPos,
                  TOperHandle2 handlePre,
                  EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_grad(std::move(grad))
        , m_hotPos(hotPos)
        , m_handlePre(std::move(handlePre))
        , m_evalOutput(std::move(evalOutput)) { }
        
    void Eval() override
    {
        const auto& grad = m_grad.Data().Value();
        const auto& p_pre = m_handlePre.Data();
        assert(p_pre.RowNum() == 1);
        
        size_t colNum = p_pre.ColNum();
        assert(m_hotPos < colNum);
        
        m_evalOutput.Allocate(1, colNum);
        auto& res = m_evalOutput.MutableData();

        auto mem_res = LowerAccess(res);
        ElementType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < colNum; ++i)
        {
            r[i] = p_pre(0, i) * grad;
        }
        r[m_hotPos] -= grad;
        m_evalOutput.SetEval();
    }

private:
    TOperHandle1 m_grad;
    size_t m_hotPos;
    TOperHandle2 m_handlePre;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

template <typename TOperHandle1, typename TOperHandle2, typename TOperHandle3,
          typename TElem, typename TDevice, typename TCate>
class Case2EvalUnit;

template <typename TOperHandle1, typename TOperHandle2, typename TOperHandle3, typename TElem>
class Case2EvalUnit<TOperHandle1, TOperHandle2, TOperHandle3, TElem, DeviceTags::CPU, CategoryTags::Matrix>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;

    Case2EvalUnit(TOperHandle1 grad,
                  TOperHandle2 handleTar,
                  TOperHandle3 handlePre,
                  EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_grad(grad)
        , m_handleTar(std::move(handleTar))
        , m_handlePre(std::move(handlePre))
        , m_evalOutput(std::move(evalOutput)) { }

    void Eval() override
    {
        const auto& grad = m_grad.Data().Value();
        const auto& p_tar = m_handleTar.Data();
        const auto& p_pre = m_handlePre.Data();

        assert(p_tar.RowNum() == 1);
        assert(p_pre.RowNum() == 1);

        size_t colNum = p_tar.ColNum();
        assert(colNum == p_pre.ColNum());
        
        m_evalOutput.Allocate(1, colNum);
        auto& res = m_evalOutput.MutableData();

        ElementType sum = ElementType();
        for (size_t i = 0; i < colNum; ++i)
        {
            sum += p_tar(0, i);
        }

        auto mem_res = LowerAccess(res);
        ElementType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < colNum; ++i)
        {
            r[i] = (p_pre(0, i) * sum - p_tar(0, i)) * grad;
        }
        m_evalOutput.SetEval();
    }

private:
    TOperHandle1 m_grad;
    TOperHandle2 m_handleTar;
    TOperHandle3 m_handlePre;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

template <typename TOperHandle1, typename TOperHandle2, typename TOperHandle3, typename TElem>
class Case2EvalUnit<TOperHandle1, TOperHandle2, TOperHandle3, TElem, DeviceTags::CPU, CategoryTags::BatchMatrix>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;

    Case2EvalUnit(TOperHandle1 grad,
                  TOperHandle2 handleTar,
                  TOperHandle3 handlePre,
                  EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> evalOutput)
        : m_grad(grad)
        , m_handleTar(std::move(handleTar))
        , m_handlePre(std::move(handlePre))
        , m_evalOutput(std::move(evalOutput)) { }

    void Eval() override
    {
        const auto& grad = m_grad.Data();
        const auto& p_tar = m_handleTar.Data();
        const auto& p_pre = m_handlePre.Data();

        assert(p_tar.RowNum() == 1);
        assert(p_pre.RowNum() == 1);

        size_t colNum = p_tar.ColNum();
        size_t batchNum = p_tar.BatchNum();
        assert(colNum == p_pre.ColNum());
        assert(batchNum == p_pre.BatchNum());
        
        m_evalOutput.Allocate(batchNum, 1, colNum);
        auto& res = m_evalOutput.MutableData();

        for (size_t curBatch = 0; curBatch < batchNum; ++curBatch)
        {
            ElementType sum = ElementType();
            for (size_t i = 0; i < colNum; ++i)
            {
                sum += p_tar[curBatch](0, i);
            }

            auto mem_res = LowerAccess(res[curBatch]);
            ElementType* r = mem_res.MutableRawMemory();

            for (size_t i = 0; i < colNum; ++i)
            {
                r[i] = (p_pre[curBatch](0, i) * sum - p_tar[curBatch](0, i)) * grad[curBatch];
            }
        }
        m_evalOutput.SetEval();
    }

private:
    TOperHandle1 m_grad;
    TOperHandle2 m_handleTar;
    TOperHandle3 m_handlePre;
    EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalOutput;
};

struct Calculator
{
    template <typename TEvalRes, typename TOperator1, typename TOutHandle, typename TSoftmaxResHandle>
    class Helper
    {
        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;
        using CategoryType = DataCategory<typename TEvalRes::DataType>;
        
    public:
        Helper(TEvalRes& evalRes, const TOperator1& oper1,
               TOutHandle outHandle, TSoftmaxResHandle softmaxRes)
            : m_evalRes(evalRes)
            , m_oper1(oper1)
            , m_outHandle(std::move(outHandle))
            , m_dataPtr(m_outHandle.DataPtr())
            , m_softmaxRes(std::move(softmaxRes))
        {}
        
        void Eval()
        {
            if constexpr ((IsDynamic<decltype(m_oper1.Operand2())>) && (IsMatrix<typename TEvalRes::DataType>))
            {
                if (auto ptr = m_oper1.Operand2().template TypeCast<OneHotVector<ElementType, DeviceType>>())
                {
                    auto handle1 = m_oper1.Operand1().EvalRegister();
                    using EvalUnit = Case1EvalUnit<decltype(handle1), TSoftmaxResHandle,
                                                   ElementType, DeviceType>;
                    using GroupType = TrivalEvalGroup<EvalUnit>;

                    auto dataPtr = m_outHandle.DataPtr();
                    auto depVec = {handle1.DataPtr(), m_softmaxRes.DataPtr()};
                    EvalUnit unit(std::move(handle1), ptr->HotPos(), std::move(m_softmaxRes), std::move(m_outHandle));
                    EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, std::move(depVec));
                    return;
                }
                else
                    return GenCase();
            }
            else
                return GenCase();
        }
    private:
        void GenCase()
        {
            auto handle1 = m_oper1.Operand1().EvalRegister();
            auto handle2 = m_oper1.Operand2().EvalRegister();

            using EvalUnit = Case2EvalUnit<decltype(handle1), decltype(handle2), TSoftmaxResHandle,
                                           ElementType, DeviceType, CategoryType>;
            using GroupType = TrivalEvalGroup<EvalUnit>;

            auto dataPtr = m_outHandle.DataPtr();
            auto depVec = {handle1.DataPtr(), handle2.DataPtr(), m_softmaxRes.DataPtr()};
            EvalUnit unit(std::move(handle1), std::move(handle2), std::move(m_softmaxRes), std::move(m_outHandle));
            EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, std::move(depVec));
        }
        
    private:
        TEvalRes& m_evalRes;
        const TOperator1& m_oper1;
        TOutHandle m_outHandle;
        const void* m_dataPtr;
        TSoftmaxResHandle m_softmaxRes;
    };
    
    
    template <typename TCaseRem, typename TEvalRes, typename TOper>
    static void EvalRegister(TEvalRes& evalRes, const TOper& oper)
    {
        using TOperator1 = RemConstRef<decltype(oper.Operand1())>;
        using TOperator2 = RemConstRef<decltype(oper.Operand2())>;
        if constexpr (!Valid<TOperator1, TOperator2>)
        {
            using THead = SeqHead<TCaseRem>;
            using TTail = SeqTail<TCaseRem>;
            THead::template EvalRegister<TTail>(evalRes, oper);
        }
        else
        {
            const auto& oper1 = oper.Operand1();
            const auto& oper2 = oper.Operand2();

            const auto& softmax_res = oper1.Operand3();
            if (softmax_res != oper2)
            {
                using THead = SeqHead<TCaseRem>;
                using TTail = SeqTail<TCaseRem>;
                THead::template EvalRegister<TTail>(evalRes, oper);
                return;
            }
        
            Helper helper(evalRes, oper1, evalRes.Handle(), oper2.EvalRegister());
            helper.Eval();
        }
    }
};
}
}
*/

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <MetaNN/operators/facilities/tags.h>
#include <MetaNN/operators/facilities/operator_frame.h>
#include <stdexcept>

namespace MetaNN
{
namespace OperSoftmaxGrad::NSCaseGen
{
template <typename TGradHandle, typename TInputHandle, typename TOutputHandle>
class EvalUnit : public BaseEvalUnit<DeviceTypeFromHandle<TOutputHandle>>
{
    using ElementType = ElementTypeFromHandle<TOutputHandle>;
public:
    EvalUnit(TGradHandle gradHandle, TInputHandle inputHandle, TOutputHandle outputHandle)
        : m_gradHandle(std::move(gradHandle))
        , m_inputHandle(std::move(inputHandle))
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        const auto& grad = m_gradHandle.Data();
        const auto& in = m_inputHandle.Data();
        assert(grad.Shape() == in.Shape());
        
        m_outputHandle.Allocate(grad.Shape());
        auto& out = m_outputHandle.MutableData();
        
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
        m_outputHandle.SetEval();
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

template <typename TOperandGrad, typename TOperandInput>
constexpr bool IsValidOper<OpTags::SoftmaxGrad, TOperandGrad, TOperandInput> =
    (IsMatrix<TOperandGrad> && IsMatrix<TOperandInput>) ||
    (IsBatchMatrix<TOperandGrad> && IsBatchMatrix<TOperandInput>);

template <>
struct OperSeq_<OpTags::SoftmaxGrad>
{
    using type = OperSeqContainer<TailCalculator<OperSoftmaxGrad::NSCaseGen::EvalUnit>>;
};

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