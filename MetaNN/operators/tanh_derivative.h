#pragma once

#include <type_traits>
#include <MetaNN/operators/operators.h>
#include <cmath>

namespace MetaNN
{
namespace NSTanhDerivative
{
namespace NSCaseGen
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
        , m_evalOutput(std::move(evalOutput)) { }

    void Eval() override
    {
        const auto& p_grad = m_oper1.Data();
        const auto& p_out = m_oper2.Data();

        const size_t rowNum = p_grad.RowNum();
        const size_t colNum = p_grad.ColNum();
        assert(p_out.RowNum() == rowNum);
        assert(p_out.ColNum() == colNum);

        m_evalOutput.Allocate(rowNum, colNum);
        auto& res = m_evalOutput.MutableData();

        auto mem_grad = LowerAccess(p_grad);
        auto mem_out = LowerAccess(p_out);
        auto mem_res = LowerAccess(res);

        const ElementType* r1 = mem_grad.RawMemory();
        const ElementType* r2 = mem_out.RawMemory();
        ElementType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < rowNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                r[j] = r1[j] * (1 - r2[j] * r2[j]);
            }
            r1 += colNum;
            r2 += colNum;
            r += colNum;
        }
        m_evalOutput.SetEval();
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
        const auto& p_out = m_oper2.Data();

        const size_t rowNum = p_grad.RowNum();
        const size_t colNum = p_grad.ColNum();
        const size_t batchNum = p_grad.BatchNum();
        
        assert(p_out.RowNum() == rowNum);
        assert(p_out.ColNum() == colNum);
        assert(p_out.BatchNum() == batchNum);

        m_evalOutput.Allocate(batchNum, rowNum, colNum);
        auto& res = m_evalOutput.MutableData();

        for (size_t curBatch = 0; curBatch < batchNum; ++curBatch)
        {
            auto mem_grad = LowerAccess(p_grad[curBatch]);
            auto mem_out = LowerAccess(p_out[curBatch]);
            auto mem_res = LowerAccess(res[curBatch]);

            const ElementType* r1 = mem_grad.RawMemory();
            const ElementType* r2 = mem_out.RawMemory();
            ElementType* r = mem_res.MutableRawMemory();

            for (size_t i = 0; i < rowNum; ++i)
            {
                for (size_t j = 0; j < colNum; ++j)
                {
                    r[j] = r1[j] * (1 - r2[j] * r2[j]);
                }
                r1 += colNum;
                r2 += colNum;
                r += colNum;
            }
        }
        m_evalOutput.SetEval();
    }

private:
    TOperHandle1 m_oper1;
    TOperHandle2 m_oper2;
    EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalOutput;
};

struct Calculator
{
    template <typename TCaseTail, typename TEvalRes, typename TOper>
    static void EvalRegister(TEvalRes& evalRes, const TOper& oper)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;
        using CateType = DataCategory<typename TEvalRes::DataType>;

        auto handle1 = oper.Operand1().EvalRegister();
        auto handle2 = oper.Operand2().EvalRegister();
        using UnitType = EvalUnit<decltype(handle1), decltype(handle2), ElementType, DeviceType, CateType>;
        using GroupType = TrivalEvalGroup<UnitType>;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        auto depVec = {handle1.DataPtr(), handle2.DataPtr()};
        UnitType unit(std::move(handle1), std::move(handle2), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, std::move(depVec));
    }
};
}
}

template <>
struct OperSeq_<BinaryOpTags::TanhDerivative>
{
    using type = OperSeqContainer<NSTanhDerivative::NSCaseGen::Calculator>;
};

struct OperTanhDerivative
{
    template <typename TGrad, typename TOut>
    static constexpr bool valid = (IsMatrix<TGrad> && IsMatrix<TOut>) ||
                                  (IsBatchMatrix<TGrad> && IsBatchMatrix<TOut>);
    
    template <typename TGrad, typename TOut>                              
    static auto Eval(TGrad&& p_grad, TOut&& p_out)
    {
        using ResType = BinaryOp<BinaryOpTags::TanhDerivative,
                                 RemConstRef<TGrad>,
                                 RemConstRef<TOut>>;
        return ResType(std::forward<TGrad>(p_grad), std::forward<TOut>(p_out));
    }
};

template <typename TGrad, typename TOut,
          std::enable_if_t<OperTanhDerivative::valid<TGrad, TOut>>* = nullptr>
auto TanhDerivative(TGrad&& p_grad, TOut&& p_out)
{
    return OperTanhDerivative::Eval(std::forward<TGrad>(p_grad),
                                    std::forward<TOut>(p_out));
}
}
