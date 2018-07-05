#pragma once

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
    
    
    template <typename TCaseRem, typename TEvalRes, typename TOperator1, typename TOperator2>
    static void EvalRegister(TEvalRes& evalRes, const TOperator1& oper1, const TOperator2& oper2)
    {
        if constexpr (!Valid<TOperator1, TOperator2>)
        {
            using THead = SeqHead<TCaseRem>;
            using TTail = SeqTail<TCaseRem>;
            THead::template EvalRegister<TTail>(evalRes, oper1, oper2);
        }
        else
        {
            const auto& softmax_res = oper1.Operand3();
            if (softmax_res != oper2)
            {
                using THead = SeqHead<TCaseRem>;
                using TTail = SeqTail<TCaseRem>;
                THead::template EvalRegister<TTail>(evalRes, oper1, oper2);
                return;
            }
        
            Helper helper(evalRes, oper1, evalRes.Handle(), oper2.EvalRegister());
            helper.Eval();
        }
    }
};
}

namespace CaseGen
{
template <typename TOperHandle1, typename TOperHandle2, typename TElem, typename TDevice, typename TCate>
class EvalUnit;

template <typename TOperHandle1, typename TOperHandle2, typename TElem>
class EvalUnit<TOperHandle1, TOperHandle2, TElem, DeviceTags::CPU, CategoryTags::Matrix>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;

    EvalUnit(TOperHandle1 oper1,
             TOperHandle2 oper2,
             EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_oper1(std::move(oper1))
        , m_oper2(std::move(oper2))
        , m_evalOutput(evalOutput) { }

    void Eval() override
    {
        const auto& p_grad = m_oper1.Data();
        const auto& p_sout = m_oper2.Data();
        size_t colNum = p_grad.ColNum();
        assert(p_grad.RowNum() == 1);
        assert(p_sout.RowNum() == 1);
        assert(colNum == p_sout.ColNum());
                
        Matrix<ElementType, DeviceType> tmp(colNum, colNum);
        for (size_t i = 0; i < colNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                tmp.SetValue(i, j, -1 * p_sout(0, i) * p_sout(0, j));
            }
            auto value = tmp(i, i);
            tmp.SetValue(i, i, p_sout(0, i) + value);
        }

        auto tempHandle = tmp.EvalRegister();
        using EvalUnit = NSDot::NSCaseGen::EvalUnit<decltype(m_oper1), decltype(tempHandle), ElementType, DeviceType,
                                                    CategoryTags::Matrix>;
        using GroupType = TrivalEvalGroup<EvalUnit>;

        const void* dataPtr = m_evalOutput.DataPtr();
        auto depVec = {m_oper1.DataPtr(), tempHandle.DataPtr()};
        EvalUnit unit(m_oper1, std::move(tempHandle), std::move(m_evalOutput));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, std::move(depVec));
    }

private:
    TOperHandle1 m_oper1;
    TOperHandle2 m_oper2;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

template <typename TOperHandle1, typename TOperHandle2, typename TElem>
class EvalUnit<TOperHandle1, TOperHandle2, TElem, DeviceTags::CPU, CategoryTags::BatchMatrix>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;

    EvalUnit(TOperHandle1 oper1,
             TOperHandle2 oper2,
             EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> evalOutput)
        : m_oper1(std::move(oper1))
        , m_oper2(std::move(oper2))
        , m_evalOutput(std::move(evalOutput)) { }

    void Eval() override
    {
        const auto& p_grad = m_oper1.Data();
        const auto& p_sout = m_oper2.Data();
        size_t colNum = p_grad.ColNum();
        size_t batchNum = p_grad.BatchNum();
        
        assert(p_grad.RowNum() == 1);
        assert(p_sout.RowNum() == 1);
        assert(colNum == p_sout.ColNum());
        assert(batchNum == p_sout.BatchNum());
                
        Batch<ElementType, DeviceType, CategoryTags::Matrix> tmp(batchNum, colNum, colNum);
        for (size_t curBatch = 0; curBatch < batchNum; ++curBatch)
        {
            for (size_t i = 0; i < colNum; ++i)
            {
                for (size_t j = 0; j < colNum; ++j)
                {
                    tmp.SetValue(curBatch, i, j, -1 * p_sout[curBatch](0, i) * p_sout[curBatch](0, j));
                }
                auto value = tmp[curBatch](i, i);
                tmp.SetValue(curBatch, i, i, p_sout[curBatch](0, i) + value);
            }
        }

        auto tempHandle = tmp.EvalRegister();
        using EvalUnit = NSDot::NSCaseGen::EvalUnit<decltype(m_oper1), decltype(tempHandle), ElementType, DeviceType,
                                                    CategoryTags::BatchMatrix>;
        using GroupType = TrivalEvalGroup<EvalUnit>;

        const void* dataPtr = m_evalOutput.DataPtr();
        auto depVec = {m_oper1.DataPtr(), tempHandle.DataPtr()};
        EvalUnit unit(m_oper1, std::move(tempHandle), std::move(m_evalOutput));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, std::move(depVec));
    }

private:
    TOperHandle1 m_oper1;
    TOperHandle2 m_oper2;
    EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalOutput;
};

struct Calculator
{
    template <typename TCaseTail, typename TEvalRes, typename TOperator1, typename TOperator2>
    static void EvalRegister(TEvalRes& evalRes, const TOperator1& oper1, const TOperator2& oper2)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        auto handle1 = oper1.EvalRegister();
        auto handle2 = oper2.EvalRegister();

        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;
        using CategoryType = DataCategory<typename TEvalRes::DataType>;

        using EvalUnit = EvalUnit<decltype(handle1), decltype(handle2), ElementType, DeviceType, CategoryType>;
        using GroupType = TrivalEvalGroup<EvalUnit>;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        auto depVec = {handle1.DataPtr(), handle2.DataPtr()};
        EvalUnit unit(std::move(handle1), std::move(handle2), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, std::move(depVec));
    }
};
}
}

template <>
struct OperSeq_<BinaryOpTags::VecSoftmaxDerivative>
{
    using type = OperSeqContainer<NSVecSoftmaxDerivative::CaseNLL::Calculator,
                                  NSVecSoftmaxDerivative::CaseGen::Calculator>;
};

template <typename TGrad, typename TSOut>
struct OperVecSoftmaxDerivative_
{
// valid check
private:
    using rawGrad = RemConstRef<TGrad>;
    using rawSOut = RemConstRef<TSOut>;

public:
    static constexpr bool valid = (IsMatrix<rawGrad> && IsMatrix<rawSOut>) ||
                                  (IsBatchMatrix<rawGrad> && IsBatchMatrix<rawSOut>);

public:
    static auto Eval(TGrad&& p_grad, TSOut&& p_sout)
    {
        static_assert(std::is_same<typename rawGrad::ElementType, typename rawSOut::ElementType>::value,
                      "Element type mismatch.");
        static_assert(std::is_same<typename rawGrad::DeviceType, typename rawSOut::DeviceType>::value,
                      "Device type mismatch.");

        using ResType = BinaryOp<BinaryOpTags::VecSoftmaxDerivative, rawGrad, rawSOut>;
        return ResType(std::forward<TGrad>(p_grad), std::forward<TSOut>(p_sout));
    }
};

template <typename TGrad, typename TSOut,
          std::enable_if_t<OperVecSoftmaxDerivative_<TGrad, TSOut>::valid>* = nullptr>
auto VecSoftmaxDerivative(TGrad&& p_grad, TSOut&& p_sout)
{
    return OperVecSoftmaxDerivative_<TGrad, TSOut>::Eval(std::forward<TGrad>(p_grad),
                                                         std::forward<TSOut>(p_sout));
}
}
