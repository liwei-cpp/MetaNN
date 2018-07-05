#pragma once

#include <MetaNN/operators/facilities/organizer.h>
#include <type_traits>
#include <vector>
#include <cmath>

namespace MetaNN
{
template <>
struct OperCategory_<TernaryOpTags::NegativeLogLikelihoodDerivative,
                     CategoryTags::Scalar, CategoryTags::Matrix, CategoryTags::Matrix>
{
    using type = CategoryTags::Matrix;
};

template <>
struct OperCategory_<TernaryOpTags::NegativeLogLikelihoodDerivative,
                     CategoryTags::BatchScalar, CategoryTags::BatchMatrix, CategoryTags::BatchMatrix>
{
    using type = CategoryTags::BatchMatrix;
};

template <>
class OperOrganizer<TernaryOpTags::NegativeLogLikelihoodDerivative, CategoryTags::Matrix>
{
public:
    template <typename TD1, typename TD2, typename TD3>
    OperOrganizer(const TD1& data1, const TD2& data2, const TD3& data3)
        : m_rowNum(data2.RowNum())
        , m_colNum(data2.ColNum())
    {
        assert(data2.RowNum() == data3.RowNum());
        assert(data2.ColNum() == data3.ColNum());
    }

    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }

private:
    size_t m_rowNum;
    size_t m_colNum;
};

template <>
class OperOrganizer<TernaryOpTags::NegativeLogLikelihoodDerivative, CategoryTags::BatchMatrix>
{
public:
    template <typename TD1, typename TD2, typename TD3>
    OperOrganizer(const TD1& data1, const TD2& data2, const TD3& data3)
        : m_rowNum(data2.RowNum())
        , m_colNum(data2.ColNum())
        , m_batchNum(data2.BatchNum())
    {
        assert(data2.RowNum() == data3.RowNum());
        assert(data2.ColNum() == data3.ColNum());
        assert(data2.BatchNum() == data3.BatchNum());
    }

    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }
    size_t BatchNum() const { return m_batchNum; }

private:
    size_t m_rowNum;
    size_t m_colNum;
    size_t m_batchNum;
};

template <typename TOp1, typename TOp2, typename TOp3>
struct OperElementType_<TernaryOpTags::NegativeLogLikelihoodDerivative,
                        TOp1, TOp2, TOp3>
{
    using type = typename TOp2::ElementType;
};

template <typename TOp1, typename TOp2, typename TOp3>
struct OperDeviceType_<TernaryOpTags::NegativeLogLikelihoodDerivative,
                       TOp1, TOp2, TOp3>
{
    using type = typename TOp2::DeviceType;
};

namespace NSNegativeLogLikelihoodDerivative
{
namespace NSCaseGen
{
template <typename TOperHandle1, typename TOperHandle2, typename TOperHandle3, typename TElem, typename TDevice, typename TCate>
class EvalUnit;

template <typename TOperHandle1, typename TOperHandle2, typename TOperHandle3, typename TElem>
class EvalUnit<TOperHandle1, TOperHandle2, TOperHandle3, TElem, DeviceTags::CPU, CategoryTags::Matrix>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = typename DeviceTags::CPU;

public:
    EvalUnit(TOperHandle1 grad, TOperHandle2 operTar, TOperHandle3 operPre,
             EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_grad(std::move(grad))
        , m_handleTar(std::move(operTar))
        , m_handlePre(std::move(operPre))
        , m_evalOutput(std::move(evalOutput)) {}

    void Eval() override
    {
        const auto& p_tar = m_handleTar.Data();
        const auto& p_pre = m_handlePre.Data();
        const auto& p_grad = m_grad.Data();

        const size_t rowNum = p_tar.RowNum();
        const size_t colNum = p_tar.ColNum();
        assert(p_pre.RowNum() == rowNum);
        assert(p_pre.ColNum() == colNum);
        
        m_evalOutput.Allocate(rowNum, colNum);
        auto& res = m_evalOutput.MutableData();
        
        assert(res.RowNum() == rowNum);
        assert(res.ColNum() == colNum);

        auto mem_v1 = LowerAccess(p_tar);
        auto mem_v2 = LowerAccess(p_pre);
        auto mem_res = LowerAccess(res);

        const size_t src1PackNum = mem_v1.RowLen();
        const size_t src2PackNum = mem_v2.RowLen();
        const size_t tgtPackNum = mem_res.RowLen();

        const ElementType* r1 = mem_v1.RawMemory();
        const ElementType* r2 = mem_v2.RawMemory();
        ElementType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < rowNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                r[j] = p_grad.Value() * (-r1[j] / r2[j]);
            }
            r1 += src1PackNum;
            r2 += src2PackNum;
            r += tgtPackNum;
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
class EvalUnit<TOperHandle1, TOperHandle2, TOperHandle3, TElem, DeviceTags::CPU, CategoryTags::BatchMatrix>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = typename DeviceTags::CPU;

public:
    EvalUnit(TOperHandle1 grad, TOperHandle2 operTar, TOperHandle3 operPre,
             EvalHandle<Batch<ElementType, DeviceType, CategoryTags::Matrix>> evalOutput)
        : m_grad(std::move(grad))
        , m_handleTar(std::move(operTar))
        , m_handlePre(std::move(operPre))
        , m_evalOutput(std::move(evalOutput)) {}

    void Eval() override
    {
        const auto& p_tar = m_handleTar.Data();
        const auto& p_pre = m_handlePre.Data();
        const auto& p_grad = m_grad.Data();

        const size_t rowNum = p_tar.RowNum();
        const size_t colNum = p_tar.ColNum();
        const size_t batchNum = p_tar.BatchNum();
        assert(p_pre.RowNum() == rowNum);
        assert(p_pre.ColNum() == colNum);
        assert(p_pre.BatchNum() == batchNum);
        assert(p_grad.BatchNum() == batchNum);
        
        m_evalOutput.Allocate(batchNum, rowNum, colNum);
        auto& res = m_evalOutput.MutableData();
        
        assert(res.RowNum() == rowNum);
        assert(res.ColNum() == colNum);
        assert(res.BatchNum() == batchNum);

        for (size_t curBatch = 0; curBatch < batchNum; ++curBatch)
        {
            auto mem_v1 = LowerAccess(p_tar[curBatch]);
            auto mem_v2 = LowerAccess(p_pre[curBatch]);
            auto mem_res = LowerAccess(res[curBatch]);

            const size_t src1PackNum = mem_v1.RowLen();
            const size_t src2PackNum = mem_v2.RowLen();
            const size_t tgtPackNum = mem_res.RowLen();

            const ElementType* r1 = mem_v1.RawMemory();
            const ElementType* r2 = mem_v2.RawMemory();
            ElementType* r = mem_res.MutableRawMemory();

            for (size_t i = 0; i < rowNum; ++i)
            {
                for (size_t j = 0; j < colNum; ++j)
                {
                    r[j] = p_grad[curBatch] * (-r1[j] / r2[j]);
                }
                r1 += src1PackNum;
                r2 += src2PackNum;
                r += tgtPackNum;
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
    template <typename TCaseTail, typename TEvalRes, typename TOperator1,
              typename TOperator2, typename TOperator3>
    static void EvalRegister(TEvalRes& evalRes,
                             const TOperator1& oper1, const TOperator2& oper2, const TOperator3& oper3)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;
        using CategoryType = DataCategory<typename TEvalRes::DataType>;

        auto handle1 = oper1.EvalRegister();
        auto handle2 = oper2.EvalRegister();
        auto handle3 = oper3.EvalRegister();
        using UnitType = EvalUnit<decltype(handle1), decltype(handle2), decltype(handle3),
                                  ElementType, DeviceType, CategoryType>;
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
struct OperSeq_<TernaryOpTags::NegativeLogLikelihoodDerivative>
{
    using type = OperSeqContainer<NSNegativeLogLikelihoodDerivative::NSCaseGen::Calculator>;
};

template <typename TGrad, typename TP1, typename TP2>
struct OperNegativeLogLikelihoodDerivative_
{
// valid check
private:
    using rawGrad = RemConstRef<TGrad>;
    using rawM1 = RemConstRef<TP1>;
    using rawM2 = RemConstRef<TP2>;

public:
    static constexpr bool valid = (IsScalar<rawGrad> && IsMatrix<rawM1> && IsMatrix<rawM2>) ||
                                  (IsBatchScalar<rawGrad> && IsBatchMatrix<rawM1> && IsBatchMatrix<rawM2>);

public:
    static auto Eval(TGrad&& p_grad, TP1&& p_m1, TP2&& p_m2)
    {
        static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                      "Matrices with different element types cannot do NegativeLogLikelihood derivative directly");
        static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                      "Matrices with different device types cannot do NegativeLogLikelihood derivative directly");

        using ResType = TernaryOp<TernaryOpTags::NegativeLogLikelihoodDerivative,
                                  rawGrad, rawM1, rawM2>;
        return ResType(std::forward<TGrad>(p_grad), std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
    }
};

template <typename TGrad, typename TP1, typename TP2,
          std::enable_if_t<OperNegativeLogLikelihoodDerivative_<TGrad, TP1, TP2>::valid>* = nullptr>
auto NegativeLogLikelihoodDerivative(TGrad&& p_grad, TP1&& p_tar, TP2&& p_pre)
{
    return OperNegativeLogLikelihoodDerivative_<TGrad, TP1, TP2>
                ::Eval(std::forward<TGrad>(p_grad), std::forward<TP1>(p_tar), std::forward<TP2>(p_pre));
}
}
