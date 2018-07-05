#pragma once

namespace MetaNN
{
template <>
class OperOrganizer<BinaryOpTags::Dot, CategoryTags::Matrix>
{
public:
    template <typename TD1, typename TD2>
    OperOrganizer(const TD1& data1, const TD2& data2)
        : m_rowNum(data1.RowNum())
        , m_colNum(data2.ColNum())
    {
        assert(data1.ColNum() == data2.RowNum());
    }

    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }

private:
    size_t m_rowNum;
    size_t m_colNum;
};

template <>
class OperOrganizer<BinaryOpTags::Dot, CategoryTags::BatchMatrix>
{
public:
    template <typename TD1, typename TD2>
    OperOrganizer(const TD1& data1, const TD2& data2)
        : m_rowNum(data1.RowNum())
        , m_colNum(data2.ColNum())
        , m_batchNum(data1.BatchNum())
    {
        assert(data1.ColNum() == data2.RowNum());
        assert(data1.BatchNum() == data2.BatchNum());
    }

    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }
    size_t BatchNum() const { return m_batchNum; }

private:
    size_t m_rowNum;
    size_t m_colNum;
    size_t m_batchNum;
};

namespace NSDot
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
        const size_t colNum = p_v2.ColNum();
        const size_t midNum = p_v1.ColNum();
        assert(p_v2.RowNum() == midNum);
        
        m_evalOutput.Allocate(rowNum, colNum);
        auto& res = m_evalOutput.MutableData();
        
        auto mem_res = LowerAccess(res);
        const size_t tgtPackNum = mem_res.RowLen();
        
        TElem* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < rowNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                *r = TElem();
                for (size_t k = 0; k < midNum; ++k)
                {
                    *r += p_v1(i, k) * p_v2(k, j);
                }
                ++r;
            }
            r += tgtPackNum - colNum;
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
        const size_t colNum = p_v2.ColNum();
        const size_t midNum = p_v1.ColNum();
        const size_t batchNum = p_v1.BatchNum();
        
        assert(p_v2.RowNum() == midNum);
        assert(p_v2.BatchNum() == batchNum);
        
        m_evalOutput.Allocate(batchNum, rowNum, colNum);
        auto& res = m_evalOutput.MutableData();
        
        for (size_t cur_batch = 0; cur_batch < batchNum; ++cur_batch)
        {
            auto mem_res = LowerAccess(res[cur_batch]);
            const size_t tgtPackNum = mem_res.RowLen();
        
            TElem* r = mem_res.MutableRawMemory();
            auto cur_v1 = p_v1[cur_batch];
            auto cur_v2 = p_v2[cur_batch];

            for (size_t i = 0; i < rowNum; ++i)
            {
                for (size_t j = 0; j < colNum; ++j)
                {
                    *r = TElem();
                    for (size_t k = 0; k < midNum; ++k)
                    {
                        *r += cur_v1(i, k) * cur_v2(k, j);
                    }
                    ++r;
                }
                r += tgtPackNum - colNum;
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
struct OperSeq_<BinaryOpTags::Dot>
{
    using type = OperSeqContainer<NSDot::NSCaseGen::Calculator>;
};

template <typename TP1, typename TP2>
struct OperDot_
{
// valid check
private:
    using rawM1 = RemConstRef<TP1>;
    using rawM2 = RemConstRef<TP2>;

public:
    static constexpr bool valid = (IsMatrix<rawM1> && IsMatrix<rawM2>) ||
                                  (IsBatchMatrix<rawM1> && IsMatrix<rawM2>) ||
                                  (IsMatrix<rawM1> && IsBatchMatrix<rawM2>) ||
                                  (IsBatchMatrix<rawM1> && IsBatchMatrix<rawM2>);

public:
    template <typename T1, typename T2,
              std::enable_if_t<std::is_same<T1, T2>::value>* = nullptr>
    static auto Eval(TP1&& p_m1, TP2&& p_m2)
    {
        static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                      "Matrices with different element types cannot dot directly");
        static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                      "Matrices with different device types cannot dot directly");

        using ResType = BinaryOp<BinaryOpTags::Dot, rawM1, rawM2>;
        return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
    }
    
    template <typename T1, typename T2,
              std::enable_if_t<std::is_same<CategoryTags::BatchMatrix, T1>::value>* = nullptr,
              std::enable_if_t<std::is_same<CategoryTags::Matrix, T2>::value>* = nullptr>
    static auto Eval(TP1&& p_m1, TP2&& p_m2)
    {
        static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                      "Matrices with different element types cannot dot directly");
        static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                      "Matrices with different device types cannot dot directly");
                      
        Duplicate<rawM2> tmp(std::forward<TP2>(p_m2), p_m1.BatchNum());
        using ResType = BinaryOp<BinaryOpTags::Dot, rawM1, Duplicate<rawM2>>;
        return ResType(std::forward<TP1>(p_m1), std::move(tmp));
    }
    
    template <typename T1, typename T2,
              std::enable_if_t<std::is_same<CategoryTags::Matrix, T1>::value>* = nullptr,
              std::enable_if_t<std::is_same<CategoryTags::BatchMatrix, T2>::value>* = nullptr>
    static auto Eval(TP1&& p_m1, TP2&& p_m2)
    {
        static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                      "Matrices with different element types cannot dot directly");
        static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                      "Matrices with different device types cannot dot directly");
                      
        Duplicate<rawM1> tmp(std::forward<TP1>(p_m1), p_m2.BatchNum());
        using ResType = BinaryOp<BinaryOpTags::Dot, Duplicate<rawM1>, rawM2>;
        return ResType(std::move(tmp), std::forward<TP2>(p_m2));
    }
};

template <typename TP1, typename TP2,
          std::enable_if_t<OperDot_<TP1, TP2>::valid>* = nullptr>
auto Dot(TP1&& p_m1, TP2&& p_m2)
{
    return OperDot_<TP1, TP2>::
            template Eval<DataCategory<TP1>, DataCategory<TP2>>(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
}
}
