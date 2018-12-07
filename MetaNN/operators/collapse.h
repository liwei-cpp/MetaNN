#pragma once

#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/operators/facilities/tags.h>
#include <MetaNN/operators/facilities/operator_frame.h>
#include <cassert>
#include <cstring>
#include <type_traits>
#include <utility>

namespace MetaNN
{
namespace OperCollapse
{
template <typename TOriData, typename TShape>
static constexpr bool valid = (std::is_same_v<TShape, Shape<CategoryTags::Scalar>>) ||

                              (IsMatrix<TOriData>                   && std::is_same_v<TShape, Shape<CategoryTags::Matrix>>) ||
                              (IsThreeDArray<TOriData>              && std::is_same_v<TShape, Shape<CategoryTags::Matrix>>) ||
                              (IsBatchMatrix<TOriData>              && std::is_same_v<TShape, Shape<CategoryTags::Matrix>>) ||
                              (IsBatchThreeDArray<TOriData>         && std::is_same_v<TShape, Shape<CategoryTags::Matrix>>) ||
                              (IsMatrixSequence<TOriData>           && std::is_same_v<TShape, Shape<CategoryTags::Matrix>>) ||
                              (IsThreeDArraySequence<TOriData>      && std::is_same_v<TShape, Shape<CategoryTags::Matrix>>) ||
                              (IsBatchMatrixSequence<TOriData>      && std::is_same_v<TShape, Shape<CategoryTags::Matrix>>) ||
                              (IsBatchThreeDArraySequence<TOriData> && std::is_same_v<TShape, Shape<CategoryTags::Matrix>>) ||

                              (IsThreeDArray<TOriData>              && std::is_same_v<TShape, Shape<CategoryTags::ThreeDArray>>) ||
                              (IsBatchThreeDArray<TOriData>         && std::is_same_v<TShape, Shape<CategoryTags::ThreeDArray>>) ||
                              (IsThreeDArraySequence<TOriData>      && std::is_same_v<TShape, Shape<CategoryTags::ThreeDArray>>) ||
                              (IsBatchThreeDArraySequence<TOriData> && std::is_same_v<TShape, Shape<CategoryTags::ThreeDArray>>);
template<typename TOriShape>
bool ShapeMatch(const TOriShape&, const Shape<CategoryTags::Scalar>&)
{
    return true;
};

template<typename TOriShape>
bool ShapeMatch(const TOriShape& ori, const Shape<CategoryTags::Matrix>& aim)
{
    return ((ori.RowNum() == aim.RowNum()) && (ori.ColNum() == aim.ColNum()));
};

template<typename TOriShape>
bool ShapeMatch(const TOriShape& ori, const Shape<CategoryTags::ThreeDArray>& aim)
{
    return ((ori.RowNum() == aim.RowNum()) &&
            (ori.ColNum() == aim.ColNum()) &&
            (ori.PageNum() == aim.PageNum()));
};

template <typename TOriData, typename TShape>
static auto CreateOpTemplate(TOriData&& data, TShape&& shape)
{
    using RawDataType = RemConstRef<TOriData>;
    using RawShapeType = RemConstRef<TShape>;
        
    using ResType = Operator<OpTags::Collapse, RawDataType, RawShapeType>;
    return ResType(std::forward<TOriData>(data),
                   std::forward<TShape>(shape));
}

template <typename TInputHandle, typename TShape, typename TOutputHandle, typename TDevice>
class EvalUnit : public BaseEvalUnit<TDevice>
{
public:
    EvalUnit(TInputHandle oriHandle, TShape shape, TOutputHandle outputHandle)
        : m_inputHandle(std::move(oriHandle))
        , m_shape(std::move(shape))
        , m_outputHandle(std::move(outputHandle))
    {}
    
    void Eval() override final
    {
        m_outputHandle.Allocate(m_shape);
        
        const auto& in = m_inputHandle.Data();
        const size_t inCount = in.Shape().Count();

        auto& out = m_outputHandle.MutableData();
        const size_t outCount = out.Shape().Count();
                
        static_assert(std::is_same_v<TDevice, DeviceTags::CPU>, "Currently only CPU is supported");
        using ElementType = ElementTypePicker<decltype(in)>;
        
        auto low_in = LowerAccess(in);
        ElementType* mem_in = low_in.MutableRawMemory();
        
        using OutputCategory = DataCategory<decltype(out)>;
        if constexpr(std::is_same_v<OutputCategory, CategoryTags::Scalar>)
        {
            ElementType elem = std::accumulate(mem_in, mem_in + inCount, ElementType{});
            out.SetValue(elem);
        }
        else
        {
            auto low_out = LowerAccess(out);
            ElementType* mem_out = low_out.MutableRawMemory();
            assert(inCount % outCount == 0);
            
            // copy the first bucket for initialization, accumulate the other parts.
            const size_t loopCount = inCount / outCount;
            memcpy(mem_out, mem_in, sizeof(ElementType) * outCount);
            mem_in += outCount;
            for (size_t i = 1; i < loopCount; ++i)
            {
                for (size_t j = 0; j < outCount; ++j)
                {
                    mem_out[j] += mem_in[j];
                }
                mem_in += outCount;
            }
        }
        m_outputHandle.SetEval();
    }
    
private:
    const TInputHandle m_inputHandle;
    const TShape m_shape;
    TOutputHandle m_outputHandle;
};

struct Calculator
{
    template <typename TEvalRes, typename TOriData, typename TShape>
    static void EvalRegister(TEvalRes& evalRes, const TOriData& oriData, const TShape& shape)
    {
        using DeviceType = typename TEvalRes::DataType::DeviceType;

        auto handle = oriData.EvalRegister();
        auto outHandle = evalRes.Handle();
        
        using UnitType = EvalUnit<decltype(handle), TShape, decltype(outHandle), DeviceType>;
        using GroupType = TrivalEvalGroup<UnitType>;

        const void* dataPtr = outHandle.DataPtr();
        const void* depVec = handle.DataPtr();
        UnitType unit(std::move(handle), shape, std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, {depVec});
    }
};
}

// Note: since operator<OpTags::Duplicate> cannot deduce its category from operands,
// so here involve a partial specification.
template <typename TOriData, typename TCategory>
class Operator<OpTags::Collapse, TOriData, Shape<TCategory>>
{
    static_assert(std::is_same_v<RemConstRef<TOriData>, TOriData>, "TOriData is not an available type.");
    
public:
    using CategoryTag = TCategory;
    using ElementType = typename TOriData::ElementType;
    using DeviceType = typename TOriData::DeviceType;
    
public:
    Operator(TOriData data, MetaNN::Shape<TCategory> shape)
        : m_oriData(std::move(data))
        , m_shape(std::move(shape))
    {}

    const auto& Shape() const noexcept
    {
        return m_shape;
    }
    
    bool operator== (const Operator& val) const
    {
        return (m_oriData == val.m_oriData) &&
               (m_shape == val.m_shape);
    }

    auto EvalRegister() const
    {
        if (!m_evalBuf.IsEvaluated())
        {
            OperCollapse::Calculator::EvalRegister(m_evalBuf, m_oriData, m_shape);
        }
        return m_evalBuf.ConstHandle();
    }

    const auto& Operand() const noexcept
    {
        return m_oriData;
    }

private:
    const TOriData m_oriData;
    const MetaNN::Shape<CategoryTag> m_shape;
    
    using TPrincipal = PrincipalDataType<CategoryTag, ElementType, DeviceType>;
    EvalBuffer<TPrincipal> m_evalBuf;
};

template <typename TOriData, typename TShape,
          std::enable_if_t<OperCollapse::valid<TOriData, RemConstRef<TShape>>>* = nullptr>
auto Collapse(TOriData&& data, TShape&& shape)
{
    using OriShape = RemConstRef<decltype(data.Shape())>;
    using NewShape = RemConstRef<TShape>;
    if constexpr (std::is_same_v<OriShape, NewShape>)
    {
        if (data.Shape() != shape)
        {
            throw std::runtime_error("Plain duplicate need identical shape.");
        }
        return std::forward<TOriData>(data);
    }
    else
    {
        if (!OperCollapse::ShapeMatch(data.Shape(), shape))
        {
            throw std::runtime_error("Cannot duplicate for un-match shape.");
        }
        return OperCollapse::CreateOpTemplate(std::forward<TOriData>(data),
                                              std::forward<TShape>(shape));
    }
}
}