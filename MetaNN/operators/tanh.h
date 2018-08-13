#pragma once

#include <type_traits>
#include <MetaNN/operators/operators.h>
#include <cmath>

namespace MetaNN
{
namespace NSTanh
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
        const size_t rowNum = p_v.RowNum();
        const size_t colNum = p_v.ColNum();
        
        m_evalOutput.Allocate(rowNum, colNum);
        auto& res = m_evalOutput.MutableData();
        
        auto mem_v1 = LowerAccess(p_v);
        auto mem_res = LowerAccess(res);

        const size_t src1PackNum = mem_v1.RowLen();
        const size_t tgtPackNum = mem_res.RowLen();

        const ElementType* r1 = mem_v1.RawMemory();
        ElementType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < rowNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                r[j] = (ElementType)(tanh(r1[j]));
            }
            r1 += src1PackNum;
            r += tgtPackNum;
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
        const size_t rowNum = p_v.RowNum();
        const size_t colNum = p_v.ColNum();
        const size_t batchNum = p_v.BatchNum();
        
        m_evalOutput.Allocate(batchNum, rowNum, colNum);
        auto& res = m_evalOutput.MutableData();
        
        for (size_t cur_batch = 0; cur_batch < batchNum; ++cur_batch)
        {
            auto mem_v1 = LowerAccess(p_v[cur_batch]);
            auto mem_res = LowerAccess(res[cur_batch]);

            const size_t src1PackNum = mem_v1.RowLen();
            const size_t tgtPackNum = mem_res.RowLen();

            const ElementType* r1 = mem_v1.RawMemory();
            ElementType* r = mem_res.MutableRawMemory();

            for (size_t i = 0; i < rowNum; ++i)
            {
                for (size_t j = 0; j < colNum; ++j)
                {
                    r[j] = (ElementType)(tanh(r1[j]));
                }
                r1 += src1PackNum;
                r += tgtPackNum;
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
    template <typename TCaseTail, typename TEvalRes, typename TOp>
    static void EvalRegister(TEvalRes& evalRes, const TOp& oper)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;
        using CategoryType = DataCategory<typename TEvalRes::DataType>;

        const auto& data = oper.Operand();
        auto handle = data.EvalRegister();
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
struct OperSeq_<UnaryOpTags::Tanh>
{
    using type = OperSeqContainer<NSTanh::NSCaseGen::Calculator>;
};

struct OperTanh
{
    template <typename T>
    static constexpr bool valid = IsMatrix<T> || IsBatchMatrix<T>;
    
    template <typename T>
    static auto Eval(T&& p_m)
    {
        using ResType = UnaryOp<UnaryOpTags::Tanh, RemConstRef<T>>;
        return ResType(std::forward<T>(p_m));
    }
};

template <typename TP,
          std::enable_if_t<OperTanh::valid<TP>>* = nullptr>
auto Tanh(TP&& p_m)
{
    return OperTanh::Eval(std::forward<TP>(p_m));
}
}
