#pragma once

namespace MetaNN
{
template <>
class OperOrganizer<UnaryOpTags::Transpose, CategoryTags::Matrix>
{
public:
    template <typename TData>
    OperOrganizer(const TData& data)
        : m_rowNum(data.ColNum())
        , m_colNum(data.RowNum())
    { }

    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }

private:
    size_t m_rowNum;
    size_t m_colNum;
};

template <>
class OperOrganizer<UnaryOpTags::Transpose, CategoryTags::BatchMatrix>
    : public OperOrganizer<UnaryOpTags::Transpose, CategoryTags::Matrix>
{
    using BaseType = OperOrganizer<UnaryOpTags::Transpose, CategoryTags::Matrix>;
public:
    template <typename TData>
    OperOrganizer(const TData& data)
        : BaseType(data)
        , m_batchNum(data.BatchNum())
    { }

    size_t BatchNum() const { return m_batchNum; }

private:
    size_t m_batchNum;
};

namespace NSTranspose
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
        
        m_evalOutput.Allocate(colNum, rowNum);
        auto& res = m_evalOutput.MutableData();

        auto mem_v1 = LowerAccess(p_v);
        const size_t src1PackNum = mem_v1.RowLen();
        const ElementType* r1 = mem_v1.RawMemory();

        auto mem_res = LowerAccess(res);
        const size_t resPackNum = mem_res.RowLen();
        ElementType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < rowNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                r[j * resPackNum + i] = r1[j];
            }
            r1 += src1PackNum;
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
        
        m_evalOutput.Allocate(batchNum, colNum, rowNum);
        auto& res = m_evalOutput.MutableData();

        for (size_t curBatch = 0; curBatch < batchNum; ++curBatch)
        {
            auto mem_v1 = LowerAccess(p_v[curBatch]);
            const size_t src1PackNum = mem_v1.RowLen();
            const ElementType* r1 = mem_v1.RawMemory();

            auto mem_res = LowerAccess(res[curBatch]);
            const size_t resPackNum = mem_res.RowLen();
            ElementType* r = mem_res.MutableRawMemory();

            for (size_t i = 0; i < rowNum; ++i)
            {
                for (size_t j = 0; j < colNum; ++j)
                {
                    r[j * resPackNum + i] = r1[j];
                }
                r1 += src1PackNum;
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
        using CateType = DataCategory<typename TEvalRes::DataType>;

        auto handle = oper.EvalRegister();
        using UnitType = EvalUnit<decltype(handle), ElementType, DeviceType, CateType>;
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
struct OperSeq_<UnaryOpTags::Transpose>
{
    using type = OperSeqContainer<NSTranspose::NSCaseGen::Calculator>;
};

template <typename TP>
struct OperTranspose_
{
// valid check
private:
    using rawM = RemConstRef<TP>;

public:
    static constexpr bool valid = IsMatrix<rawM> || IsBatchMatrix<rawM>;

public:
    static auto Eval(TP&& p_m)
    {
        using ResType = UnaryOp<UnaryOpTags::Transpose, rawM>;
        return ResType(std::forward<TP>(p_m));
    }
};

template <typename TP,
          std::enable_if_t<OperTranspose_<TP>::valid>* = nullptr>
auto Transpose(TP&& p_m)
{
    return OperTranspose_<TP>::Eval(std::forward<TP>(p_m));
}
}
