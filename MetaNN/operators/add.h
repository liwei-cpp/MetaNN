#pragma once

#include <MetaNN/data/scalar.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/matrices/trival_matrix.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/operators/facilities/tags.h>
#include <MetaNN/operators/operators.h>
#include <cassert>
#include <type_traits>
#include <utility>

namespace MetaNN
{
namespace NSAdd
{
namespace NSCaseGen
{
template <typename TOperHandle1, typename TOperHandle2, typename TElem, typename TDevice, typename TCategory>
class EvalUnit;

template <typename TOperHandle1, typename TOperHandle2, typename TElem>
class EvalUnit<TOperHandle1, TOperHandle2, TElem, DeviceTags::CPU, CategoryTags::Matrix>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    EvalUnit(TOperHandle1 oper1,
             TOperHandle2 oper2,
             EvalHandle<Matrix<TElem, DeviceTags::CPU>> evalOutput)
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
                r[j] = r1[j] + r2[j];
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
    EvalHandle<Matrix<TElem, DeviceTags::CPU>> m_evalOutput;
};

template <typename TOperHandle1, typename TOperHandle2, typename TElem>
class EvalUnit<TOperHandle1, TOperHandle2, TElem, DeviceTags::CPU, CategoryTags::BatchMatrix>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    EvalUnit(TOperHandle1 oper1,
             TOperHandle2 oper2,
             EvalHandle<Batch<TElem, DeviceTags::CPU, CategoryTags::Matrix>> evalOutput)
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
        
        for (size_t cur_bat = 0; cur_bat < batchNum; ++cur_bat)
        {
            const auto mem_v1 = LowerAccess(p_v1[cur_bat]);
            const auto mem_v2 = LowerAccess(p_v2[cur_bat]);
            auto mem_res = LowerAccess(res[cur_bat]);

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
                    r[j] = r1[j] + r2[j];
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
    EvalHandle<Batch<TElem, DeviceTags::CPU, CategoryTags::Matrix>> m_evalOutput;
};

struct Calculator
{
    template <typename TCaseTail, typename TEvalRes, typename TOperator1, typename TOperator2>
    static void EvalRegister(TEvalRes& evalRes, const TOperator1& oper1, const TOperator2& oper2)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;
        using CategoryType = DataCategory<typename TEvalRes::DataType>;

        auto handle1 = oper1.EvalRegister();
        auto handle2 = oper2.EvalRegister();
        using UnitType = EvalUnit<decltype(handle1), decltype(handle2),
                                  ElementType, DeviceType, CategoryType>;
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
struct OperSeq_<BinaryOpTags::Add>
{
    using type = OperSeqContainer<NSAdd::NSCaseGen::Calculator>;
};

template <typename TP1, typename TP2>
struct OperAdd_
{
// valid check
private:
    using rawM1 = RemConstRef<TP1>;
    using rawM2 = RemConstRef<TP2>;

public:
    static constexpr bool valid = (IsMatrix<rawM1> && IsMatrix<rawM2>) ||
                                  (IsMatrix<rawM1> && IsScalar<rawM2>) ||
                                  (IsScalar<rawM1> && IsMatrix<rawM2>) ||
                                  (IsBatchMatrix<rawM1> && IsMatrix<rawM2>) ||
                                  (IsMatrix<rawM1> && IsBatchMatrix<rawM2>) ||
                                  (IsBatchMatrix<rawM1> && IsBatchMatrix<rawM2>) ||
                                  (IsBatchMatrix<rawM1> && IsScalar<rawM2>) ||
                                  (IsScalar<rawM1> && IsBatchMatrix<rawM2>);

public:
    template <typename T1, typename T2,
              std::enable_if_t<std::is_same<T1, T2>::value>* = nullptr>
    static auto Eval(TP1&& p_m1, TP2&& p_m2)
    {
        static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                      "Matrices with different element types cannot add directly");
        static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                      "Matrices with different device types cannot add directly");

        using ResType = BinaryOp<BinaryOpTags::Add, rawM1, rawM2>;
        return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
    }

    template<typename T1, typename T2,
             std::enable_if_t<std::is_same<CategoryTags::Scalar, T1>::value>* = nullptr,
             std::enable_if_t<std::is_same<CategoryTags::Matrix, T2>::value>* = nullptr>
    static auto Eval(TP1&& p_m1, TP2&& p_m2)
    {
        using ElementType = typename rawM2::ElementType;
        using DeviceType = typename rawM2::DeviceType;
        auto tmpMatrix = MakeTrivalMatrix<ElementType, DeviceType>(p_m2.RowNum(), p_m2.ColNum(), std::forward<TP1>(p_m1));

        using ResType = BinaryOp<BinaryOpTags::Add,
                                 RemConstRef<decltype(tmpMatrix)>,
                                 rawM2>;
        return ResType(std::move(tmpMatrix), std::forward<TP2>(p_m2));
    }

    template<typename T1, typename T2,
             std::enable_if_t<std::is_same<CategoryTags::Matrix, T1>::value>* = nullptr,
             std::enable_if_t<std::is_same<CategoryTags::Scalar, T2>::value>* = nullptr>
    static auto Eval(TP1&& p_m1, TP2&& p_m2)
    {
        return OperAdd_<TP2, TP1>::
                template Eval<T2, T1>(std::forward<TP2>(p_m2), std::forward<TP1>(p_m1));
    }
    
    template <typename T1, typename T2,
              std::enable_if_t<std::is_same<CategoryTags::BatchMatrix, T1>::value>* = nullptr,
              std::enable_if_t<std::is_same<CategoryTags::Matrix, T2>::value>* = nullptr>
    static auto Eval(TP1&& p_m1, TP2&& p_m2)
    {
        static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                      "Matrices with different element types cannot add directly");
        static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                      "Matrices with different device types cannot add directly");
                      
        Duplicate<rawM2> tmp(std::forward<TP2>(p_m2), p_m1.BatchNum());

        using ResType = BinaryOp<BinaryOpTags::Add, rawM1, Duplicate<rawM2>>;
        return ResType(std::forward<TP1>(p_m1), std::move(tmp));
    }
    
    template <typename T1, typename T2,
              std::enable_if_t<std::is_same<CategoryTags::Matrix, T1>::value>* = nullptr,
              std::enable_if_t<std::is_same<CategoryTags::BatchMatrix, T2>::value>* = nullptr>
    static auto Eval(TP1&& p_m1, TP2&& p_m2)
    {
        return OperAdd_<TP2, TP1>::
                template Eval<T2, T1>(std::forward<TP2>(p_m2), std::forward<TP1>(p_m1));
    }
    
    template<typename T1, typename T2,
             std::enable_if_t<std::is_same<CategoryTags::Scalar, T1>::value>* = nullptr,
             std::enable_if_t<std::is_same<CategoryTags::BatchMatrix, T2>::value>* = nullptr>
    static auto Eval(TP1&& p_m1, TP2&& p_m2)
    {
        using ElementType = typename rawM2::ElementType;
        using DeviceType = typename rawM2::DeviceType;
        auto tmpMatrix = MakeTrivalMatrix<ElementType, DeviceType>(p_m2.RowNum(), p_m2.ColNum(), std::forward<TP1>(p_m1));
        
        auto tmpBatchMatrix = MakeDuplicate(p_m2.BatchNum(), std::move(tmpMatrix));

        using ResType = BinaryOp<BinaryOpTags::Add,
                                 RemConstRef<decltype(tmpBatchMatrix)>,
                                 rawM2>;
        return ResType(std::move(tmpBatchMatrix), std::forward<TP2>(p_m2));
    }
    
    template <typename T1, typename T2,
              std::enable_if_t<std::is_same<CategoryTags::BatchMatrix, T1>::value>* = nullptr,
              std::enable_if_t<std::is_same<CategoryTags::Scalar, T2>::value>* = nullptr>
    static auto Eval(TP1&& p_m1, TP2&& p_m2)
    {
        return OperAdd_<TP2, TP1>::
                template Eval<T2, T1>(std::forward<TP2>(p_m2), std::forward<TP1>(p_m1));
    }
};

template <typename TP1, typename TP2,
          std::enable_if_t<OperAdd_<TP1, TP2>::valid>* = nullptr>
auto operator+ (TP1&& p_m1, TP2&& p_m2)
{
    using Cate1 = DataCategory<TP1>;
    using Cate2 = DataCategory<TP2>;
    return OperAdd_<TP1, TP2>::
            template Eval<Cate1, Cate2>(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
}
}