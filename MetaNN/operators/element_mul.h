#pragma once

#include <type_traits>
#include <MetaNN/operators/operators.h>
namespace MetaNN
{
namespace NSElementMul
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
        , m_evalOutput(evalOutput) { }

    void Eval() override
    {
        const auto& p_v1 = m_oper1.Data();
        const auto& p_v2 = m_oper2.Data();
        const size_t rowNum = p_v1.RowNum();
        const size_t colNum = p_v1.ColNum();
        assert(p_v2.RowNum() == rowNum);
        assert(p_v2.ColNum() == colNum);

        m_evalOutput.Allocate(rowNum, colNum);
        auto& res = m_evalOutput.MutableData();

        const auto mem_v1 = LowerAccess(p_v1);
        const auto mem_v2 = LowerAccess(p_v2);
        auto mem_res = LowerAccess(res);

        const size_t src1PackNum = mem_v1.RowLen();
        const size_t src2PackNum = mem_v2.RowLen();
        const size_t tgtPackNum = mem_res.RowLen();

        const TElem* r1 = mem_v1.RawMemory();
        const TElem* r2 = mem_v2.RawMemory();
        TElem* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < rowNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                r[j] = r1[j] * r2[j];
            }
            r1 += src1PackNum;
            r2 += src2PackNum;
            r += tgtPackNum;
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
        , m_evalOutput(evalOutput) { }

    void Eval() override
    {
        const auto& p_v1 = m_oper1.Data();
        const auto& p_v2 = m_oper2.Data();
        const size_t rowNum = p_v1.RowNum();
        const size_t colNum = p_v1.ColNum();
        const size_t batchNum = p_v1.BatchNum();
        
        assert(p_v2.RowNum() == rowNum);
        assert(p_v2.ColNum() == colNum);
        assert(p_v2.BatchNum() == batchNum);

        m_evalOutput.Allocate(batchNum, rowNum, colNum);
        auto& res = m_evalOutput.MutableData();

        for (size_t cur_batch = 0; cur_batch < batchNum; ++cur_batch)
        {
            const auto mem_v1 = LowerAccess(p_v1[cur_batch]);
            const auto mem_v2 = LowerAccess(p_v2[cur_batch]);
            auto mem_res = LowerAccess(res[cur_batch]);

            const size_t src1PackNum = mem_v1.RowLen();
            const size_t src2PackNum = mem_v2.RowLen();
            const size_t tgtPackNum = mem_res.RowLen();

            const auto* r1 = mem_v1.RawMemory();
            const auto* r2 = mem_v2.RawMemory();
            auto* r = mem_res.MutableRawMemory();

            for (size_t i = 0; i < rowNum; ++i)
            {
                for (size_t j = 0; j < colNum; ++j)
                {
                    r[j] = r1[j] * r2[j];
                }
                r1 += src1PackNum;
                r2 += src2PackNum;
                r += tgtPackNum;
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
        using CategoryType = DataCategory<typename TEvalRes::DataType>;

        const auto& oper1 = oper.Operand1();
        const auto& oper2 = oper.Operand2();
        auto handle1 = oper1.EvalRegister();
        auto handle2 = oper2.EvalRegister();
        using UnitType = EvalUnit<decltype(handle1), decltype(handle2), ElementType, DeviceType, CategoryType>;
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
struct OperSeq_<BinaryOpTags::ElementMul>
{
    using type = OperSeqContainer<NSElementMul::NSCaseGen::Calculator>;
};

struct OperElementMul
{
    template <typename T1, typename T2>
    static constexpr bool valid = (IsMatrix<T1> && IsMatrix<T2>) ||
                                  (IsMatrix<T1> && IsScalar<T2>) ||
                                  (IsScalar<T1> && IsMatrix<T2>) ||
                                  (IsMatrix<T1> && IsBatchMatrix<T2>) ||
                                  (IsBatchMatrix<T1> && IsMatrix<T2>)  ||
                                  (IsBatchMatrix<T1> && IsBatchMatrix<T2>) ||
                                  (IsBatchMatrix<T1> && IsScalar<T2>) ||
                                  (IsScalar<T1> && IsBatchMatrix<T2>);
    
    template <typename T1, typename T2,
              std::enable_if_t<std::is_same<DataCategory<T1>,
                                            DataCategory<T2>>::value>* = nullptr>
    static auto Eval(T1&& p_m1, T2&& p_m2)
    {
        using rawM1 = RemConstRef<T1>;
        using rawM2 = RemConstRef<T2>;
        static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                      "Matrices with different element types cannot multiply directly");
        static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                      "Matrices with different device types cannot multiply directly");

        using ResType = BinaryOp<BinaryOpTags::ElementMul, rawM1, rawM2>;
        return ResType(std::forward<T1>(p_m1), std::forward<T2>(p_m2));
    }
    
    template<typename T1, typename T2,
             std::enable_if_t<IsScalar<T1>>* = nullptr,
             std::enable_if_t<IsMatrix<T2>>* = nullptr>
    static auto Eval(T1&& p_m1, T2&& p_m2)
    {
        using rawM2 = RemConstRef<T2>;
        using ElementType = typename rawM2::ElementType;
        using DeviceType = typename rawM2::DeviceType;
        auto tmpMatrix = MakeTrivalMatrix<ElementType, DeviceType>(p_m2.RowNum(), p_m2.ColNum(), std::forward<T1>(p_m1));

        using ResType = BinaryOp<BinaryOpTags::ElementMul,
                                 RemConstRef<decltype(tmpMatrix)>,
                                 rawM2>;
        return ResType(std::move(tmpMatrix), std::forward<T2>(p_m2));
    }
    
    template<typename T1, typename T2,
             std::enable_if_t<IsMatrix<T1>>* = nullptr,
             std::enable_if_t<IsScalar<T2>>* = nullptr>
    static auto Eval(T1&& p_m1, T2&& p_m2)
    {
        return Eval(std::forward<T2>(p_m2), std::forward<T1>(p_m1));
    }
    
    template <typename T1, typename T2,
              std::enable_if_t<IsMatrix<T1>>* = nullptr,
              std::enable_if_t<IsBatchMatrix<T2>>* = nullptr>
    static auto Eval(T1&& p_m1, T2&& p_m2)
    {
        using rawM1 = RemConstRef<T1>;
        using rawM2 = RemConstRef<T2>;
        static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                      "Matrices with different element types cannot multiply directly");
        static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                      "Matrices with different device types cannot multiply directly");

        Duplicate<rawM1> tmp(std::forward<T1>(p_m1), p_m2.BatchNum());
        using ResType = BinaryOp<BinaryOpTags::ElementMul, Duplicate<rawM1>, rawM2>;
        return ResType(std::move(tmp), std::forward<T2>(p_m2));
    }
    
    template <typename T1, typename T2,
              std::enable_if_t<IsBatchMatrix<T1>>* = nullptr,
              std::enable_if_t<IsMatrix<T2>>* = nullptr>
    static auto Eval(T1&& p_m1, T2&& p_m2)
    {
        return Eval(std::forward<T2>(p_m2), std::forward<T1>(p_m1));
    }
    
    template<typename T1, typename T2,
             std::enable_if_t<IsBatchMatrix<T1>>* = nullptr,
             std::enable_if_t<IsScalar<T2>>* = nullptr>
    static auto Eval(T1&& p_m1, T2&& p_m2)
    {
        using rawM1 = RemConstRef<T1>;
        using ElementType = typename rawM1::ElementType;
        using DeviceType = typename rawM1::DeviceType;
        auto tmpMatrix = MakeTrivalMatrix<ElementType, DeviceType>(p_m1.RowNum(), p_m1.ColNum(), std::forward<T2>(p_m2));
        auto tmpBatchMatrix = MakeDuplicate(p_m1.BatchNum(), std::move(tmpMatrix));
        
        using ResType = BinaryOp<BinaryOpTags::ElementMul,
                                 rawM1,
                                 RemConstRef<decltype(tmpBatchMatrix)>>;
        return ResType(std::forward<T1>(p_m1), std::move(tmpBatchMatrix));
    }
    
    template <typename T1, typename T2,
              std::enable_if_t<IsScalar<T1>>* = nullptr,
              std::enable_if_t<IsBatchMatrix<T2>>* = nullptr>
    static auto Eval(T1&& p_m1, T2&& p_m2)
    {
        return Eval(std::forward<T2>(p_m2), std::forward<T1>(p_m1));
    }
};

template <typename TP1, typename TP2,
          std::enable_if_t<OperElementMul::valid<TP1, TP2>>* = nullptr>
auto operator* (TP1&& p_m1, TP2&& p_m2)
{
    return OperElementMul::Eval(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
}
}
