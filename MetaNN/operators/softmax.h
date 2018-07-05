#pragma once

#include <type_traits>
#include <MetaNN/operators/operators.h>
#include <cmath>
#include <algorithm>

namespace MetaNN
{
namespace NSVecSoftmax
{
namespace NSCaseGen
{
template <typename TOperHandle, typename TElem, typename TDevice, typename TCate>
class EvalUnit;

template <typename TOperHandle, typename TElem>
class EvalUnit<TOperHandle, TElem, DeviceTags::CPU, CategoryTags::Matrix>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;

    EvalUnit(TOperHandle oper,
             EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_oper(std::move(oper))
        , m_evalOutput(evalOutput) { }

    void Eval() override
    {
        const auto& p_v = m_oper.Data();
        assert(p_v.RowNum() == 1);
        const size_t colNum = p_v.ColNum();
        
        m_evalOutput.Allocate(1, colNum);
        if (colNum == 0) return;
        auto& res = m_evalOutput.MutableData();

        auto mem_v1 = LowerAccess(p_v);
        auto mem_res = LowerAccess(res);

        const ElementType* r1 = mem_v1.RawMemory();
        ElementType* r = mem_res.MutableRawMemory();
        
        auto maxElem = *std::max_element(r1, r1 + colNum);

        ElementType sum = ElementType();

        for (size_t i = 0; i < colNum; ++i)
        {
            r[i] = exp(r1[i] - maxElem);
            sum += r[i];
        }

        for (size_t i = 0; i < colNum; ++i)
        {
            r[i] /= sum;
        }
        m_evalOutput.SetEval();
    }

private:
    TOperHandle m_oper;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

template <typename TOperHandle, typename TElem>
class EvalUnit<TOperHandle, TElem, DeviceTags::CPU, CategoryTags::BatchMatrix>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;

    EvalUnit(TOperHandle oper,
             EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> evalOutput)
        : m_oper(std::move(oper))
        , m_evalOutput(std::move(evalOutput)) { }

    void Eval() override
    {
        const auto& p_v = m_oper.Data();
        assert(p_v.RowNum() == 1);
        const size_t colNum = p_v.ColNum();
        const size_t batchNum = p_v.BatchNum();
        
        m_evalOutput.Allocate(batchNum, 1, colNum);
        if (colNum == 0) return;
        auto& res = m_evalOutput.MutableData();

        for (size_t curBatch = 0; curBatch < batchNum; ++curBatch)
        {
            auto mem_v1 = LowerAccess(p_v[curBatch]);
            auto mem_res = LowerAccess(res[curBatch]);

            const ElementType* r1 = mem_v1.RawMemory();
            ElementType* r = mem_res.MutableRawMemory();
        
            auto maxElem = *std::max_element(r1, r1 + colNum);

            ElementType sum = ElementType();

            for (size_t i = 0; i < colNum; ++i)
            {
                r[i] = exp(r1[i] - maxElem);
                sum += r[i];
            }

            for (size_t i = 0; i < colNum; ++i)
            {
                r[i] /= sum;
            }
        }
        m_evalOutput.SetEval();
    }

private:
    TOperHandle m_oper;
    EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalOutput;
};

struct Calculator
{
    template <typename TCaseTail, typename TEvalRes, typename TOperand>
    static void EvalRegister(TEvalRes& evalRes, const TOperand& oper)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;
        using CategoryType = DataCategory<typename TEvalRes::DataType>;

        auto handle = oper.EvalRegister();
        using UnitType = EvalUnit<decltype(handle), ElementType, DeviceType, CategoryType>;
        using GroupType = TrivalEvalGroup<UnitType>;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        const void* depVec = handle.DataPtr();
        UnitType unit(std::move(handle), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, {depVec});
    }
};
}
}

template <>
struct OperSeq_<UnaryOpTags::VecSoftmax>
{
    using type = OperSeqContainer<NSVecSoftmax::NSCaseGen::Calculator>;
};

template <typename TP>
struct OperVecSoftmax_
{
// valid check
private:
    using rawM = RemConstRef<TP>;

public:
    static constexpr bool valid = IsMatrix<rawM> || IsBatchMatrix<rawM>;

public:
    template <typename T>
    static auto Eval(TP&& p_m)
    {
        using ResType = UnaryOp<UnaryOpTags::VecSoftmax, rawM>;
        return ResType(std::forward<TP>(p_m));
    }
};

template <typename TP,
          std::enable_if_t<OperVecSoftmax_<TP>::valid>* = nullptr>
auto VecSoftmax(TP&& p_m)
{
    return OperVecSoftmax_<TP>::
            template Eval<DataCategory<TP>>(std::forward<TP>(p_m));
}
}
