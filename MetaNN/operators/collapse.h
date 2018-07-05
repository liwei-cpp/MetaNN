#pragma once

namespace MetaNN
{
template <>
struct OperCategory_<UnaryOpTags::Collapse, CategoryTags::BatchMatrix>
{
    using type = CategoryTags::Matrix;
};

namespace NSCollapse
{
namespace NSCaseGen
{
template <typename TOperand, typename TElem, typename TDevice>
class EvalUnit;

template <typename TOperand, typename TElem>
class EvalUnit<TOperand, TElem, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;

    EvalUnit(TOperand evalInput,
             EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_evalInput(std::move(evalInput))
        , m_evalOutput(std::move(evalOutput)) {}

    void Eval() override
    {
        const auto& p_v = m_evalInput.Data();

        const size_t rowNum = p_v.RowNum();
        const size_t colNum = p_v.ColNum();
        const size_t batchNum = p_v.BatchNum();
        m_evalOutput.Allocate(rowNum, colNum);
        
        auto& res = m_evalOutput.MutableData();

        for (size_t j = 0; j < rowNum; ++j)
        {
            for (size_t k = 0; k < colNum; ++k)
            {
                TElem tmp = 0;
                for (size_t i = 0; i < batchNum; ++i)
                {
                    tmp += p_v[i](j, k);
                }
                res.SetValue(j, k, tmp);
            }
        }
        m_evalOutput.SetEval();
    }

private:
    TOperand m_evalInput;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

struct Calculator
{
    template <typename TCaseTail, typename TEvalRes, typename TOperand>
    static void EvalRegister(TEvalRes& evalRes, const TOperand& operand)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;
        
        auto handle = operand.EvalRegister();
        using UnitType = EvalUnit<decltype(handle), ElementType, DeviceType>;
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
struct OperSeq_<UnaryOpTags::Collapse>
{
    using type = OperSeqContainer<NSCollapse::NSCaseGen::Calculator>;
};

template <typename TP>
struct OperCollapse_
{
// valid check
private:
    using rawM = std::decay_t<TP>;

public:
    static constexpr bool valid = IsBatchMatrix<rawM>;

public:
    static auto Eval(TP&& p_m)
    {
        using ResType = UnaryOp<UnaryOpTags::Collapse, rawM>;
        return ResType(std::forward<TP>(p_m));
    }
};

template <typename TP,
          std::enable_if_t<OperCollapse_<TP>::valid>* = nullptr>
auto Collapse(TP&& p_m)
{
    return OperCollapse_<TP>::Eval(std::forward<TP>(p_m));
}
}