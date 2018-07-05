#pragma once

namespace MetaNN
{
namespace NSInterpolate
{
namespace NSCaseGen
{
template <typename TOperHandle1, typename TOperHandle2, typename TOperHandle3,
          typename TElem, typename TDevice, typename TCate>
class EvalUnit;

template <typename TOperHandle1, typename TOperHandle2, typename TOperHandle3, typename TElem>
class EvalUnit<TOperHandle1, TOperHandle2, TOperHandle3, TElem, DeviceTags::CPU, CategoryTags::Matrix>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;

    EvalUnit(TOperHandle1 oper1, TOperHandle2 oper2, TOperHandle3 oper3,
             EvalHandle<Matrix<TElem, DeviceTags::CPU>> evalOutput)
        : m_oper1(std::move(oper1))
        , m_oper2(std::move(oper2))
        , m_oper3(std::move(oper3))
        , m_evalOutput(evalOutput)
    { }

    void Eval() override
    {
        const auto& p_v1 = m_oper1.Data();
        const auto& p_v2 = m_oper2.Data();
        const auto& p_v3 = m_oper3.Data();

        const size_t rowNum = p_v1.RowNum();
        const size_t colNum = p_v1.ColNum();
        assert(p_v2.RowNum() == rowNum);
        assert(p_v2.ColNum() == colNum);
        assert(p_v3.RowNum() == rowNum);
        assert(p_v3.ColNum() == colNum);
        
        m_evalOutput.Allocate(rowNum, colNum);
        auto& res = m_evalOutput.MutableData();

        auto mem_v1 = LowerAccess(p_v1);
        auto mem_v2 = LowerAccess(p_v2);
        auto mem_v3 = LowerAccess(p_v3);
        auto mem_res = LowerAccess(res);

        const size_t src1PackNum = mem_v1.RowLen();
        const size_t src2PackNum = mem_v2.RowLen();
        const size_t src3PackNum = mem_v3.RowLen();
        const size_t tgtPackNum = mem_res.RowLen();

        const TElem* r1 = mem_v1.RawMemory();
        const TElem* r2 = mem_v2.RawMemory();
        const TElem* r3 = mem_v3.RawMemory();
        TElem* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < rowNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                r[j] = r1[j] * r3[j] + r2[j] * (1 - r3[j]);
            }
            r1 += src1PackNum;
            r2 += src2PackNum;
            r3 += src3PackNum;
            r += tgtPackNum;
        }
        m_evalOutput.SetEval();
    }

private:
    TOperHandle1 m_oper1;
    TOperHandle2 m_oper2;
    TOperHandle3 m_oper3;
    EvalHandle<Matrix<TElem, DeviceTags::CPU>> m_evalOutput;
};

template <typename TOperHandle1, typename TOperHandle2, typename TOperHandle3, typename TElem>
class EvalUnit<TOperHandle1, TOperHandle2, TOperHandle3, TElem, DeviceTags::CPU, CategoryTags::BatchMatrix>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;

    EvalUnit(TOperHandle1 oper1, TOperHandle2 oper2, TOperHandle3 oper3,
             EvalHandle<Batch<TElem, DeviceTags::CPU, CategoryTags::Matrix>> evalOutput)
        : m_oper1(std::move(oper1))
        , m_oper2(std::move(oper2))
        , m_oper3(std::move(oper3))
        , m_evalOutput(evalOutput)
    { }

    void Eval() override
    {
        const auto& p_v1 = m_oper1.Data();
        const auto& p_v2 = m_oper2.Data();
        const auto& p_v3 = m_oper3.Data();

        const size_t rowNum = p_v1.RowNum();
        const size_t colNum = p_v1.ColNum();
        const size_t batchNum = p_v1.BatchNum();
        
        assert(p_v2.RowNum() == rowNum);
        assert(p_v2.ColNum() == colNum);
        assert(p_v2.BatchNum() == batchNum);
        assert(p_v3.RowNum() == rowNum);
        assert(p_v3.ColNum() == colNum);
        assert(p_v3.BatchNum() == batchNum);
        
        m_evalOutput.Allocate(batchNum, rowNum, colNum);
        auto& res = m_evalOutput.MutableData();
        
        for (size_t cur_batch = 0; cur_batch < batchNum; ++cur_batch)
        {
            auto mem_v1 = LowerAccess(p_v1[cur_batch]);
            auto mem_v2 = LowerAccess(p_v2[cur_batch]);
            auto mem_v3 = LowerAccess(p_v3[cur_batch]);
            auto mem_res = LowerAccess(res[cur_batch]);

            const size_t src1PackNum = mem_v1.RowLen();
            const size_t src2PackNum = mem_v2.RowLen();
            const size_t src3PackNum = mem_v3.RowLen();
            const size_t tgtPackNum = mem_res.RowLen();

            const auto* r1 = mem_v1.RawMemory();
            const auto* r2 = mem_v2.RawMemory();
            const auto* r3 = mem_v3.RawMemory();
            auto* r = mem_res.MutableRawMemory();

            for (size_t i = 0; i < rowNum; ++i)
            {
                for (size_t j = 0; j < colNum; ++j)
                {
                    r[j] = r1[j] * r3[j] + r2[j] * (1 - r3[j]);
                }
                r1 += src1PackNum;
                r2 += src2PackNum;
                r3 += src3PackNum;
                r += tgtPackNum;
            }
        }
        m_evalOutput.SetEval();
    }

private:
    TOperHandle1 m_oper1;
    TOperHandle2 m_oper2;
    TOperHandle3 m_oper3;
    EvalHandle<Batch<TElem, DeviceTags::CPU, CategoryTags::Matrix>> m_evalOutput;
};

struct Calculator
{
    template <typename TCaseTail, typename TEvalRes,
              typename TOperator1, typename TOperator2, typename TOperator3>
    static void EvalRegister(TEvalRes& evalRes, const TOperator1& oper1,
                             const TOperator2& oper2, const TOperator3& oper3)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;
        using CategoryType = DataCategory<typename TEvalRes::DataType>;

        auto handle1 = oper1.EvalRegister();
        auto handle2 = oper2.EvalRegister();
        auto handle3 = oper3.EvalRegister();
        using UnitType = EvalUnit<decltype(handle1), decltype(handle2), 
                                  decltype(handle3), ElementType, DeviceType, CategoryType>;
        using GroupType = TrivalEvalGroup<UnitType>;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        auto depVec = {handle1.DataPtr(), handle2.DataPtr(), handle3.DataPtr()};
        
        UnitType unit(std::move(handle1), std::move(handle2), std::move(handle3), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, std::move(depVec));
    }
};
}
}

template <>
struct OperSeq_<TernaryOpTags::Interpolate>
{
    using type = OperSeqContainer<NSInterpolate::NSCaseGen::Calculator>;
};

template <typename TP1, typename TP2, typename TP3>
struct OperInterpolate_
{
// valid check
private:
    using rawM1 = RemConstRef<TP1>;
    using rawM2 = RemConstRef<TP2>;
    using rawM3 = RemConstRef<TP3>;

public:
    static constexpr bool valid = (IsMatrix<rawM1> && IsMatrix<rawM2> && IsMatrix<rawM3>) ||
                                  (IsBatchMatrix<rawM1> && IsBatchMatrix<rawM2> && IsBatchMatrix<rawM3>);

public:
    template <typename T1, typename T2, typename T3,
              std::enable_if_t<std::is_same<T1, T2>::value>* = nullptr,
              std::enable_if_t<std::is_same<T2, T3>::value>* = nullptr>
    static auto Eval(TP1&& p_m1, TP2&& p_m2, TP3&& p_m3)
    {
        static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                      "Matrices with different element types cannot interpolate directly");
        static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                      "Matrices with different device types cannot interpolate directly");
                      
        static_assert(std::is_same<typename rawM1::ElementType, typename rawM3::ElementType>::value,
                      "Matrices with different element types cannot interpolate directly");
        static_assert(std::is_same<typename rawM1::DeviceType, typename rawM3::DeviceType>::value,
                      "Matrices with different device types cannot interpolate directly");

        using ResType = TernaryOp<TernaryOpTags::Interpolate, rawM1, rawM2, rawM3>;
        return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2), std::forward<TP3>(p_m3));
    }
};

template <typename TP1, typename TP2, typename TP3,
          std::enable_if_t<OperInterpolate_<TP1, TP2, TP3>::valid>* = nullptr>
auto Interpolate (TP1&& p_m1, TP2&& p_m2, TP3&& p_lambda)
{
    return OperInterpolate_<TP1, TP2, TP3>::
            template Eval<DataCategory<TP1>, DataCategory<TP2>, DataCategory<TP3>>(std::forward<TP1>(p_m1),
                                                                       std::forward<TP2>(p_m2),
                                                                       std::forward<TP3>(p_lambda));
}
}
