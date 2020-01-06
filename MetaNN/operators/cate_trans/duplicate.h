#pragma once

#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <MetaNN/operators/facilities/operator_frame.h>
#include <cassert>
#include <cstring>
#include <type_traits>
#include <utility>

namespace MetaNN::OpTags
{
    struct Duplicate;
}

namespace MetaNN
{
template <typename TOriData, typename TShape>
constexpr bool IsValidOper<OpTags::Duplicate, TOriData, TShape>
    = (IsScalar<TOriData>) ||

      (IsMatrix<TOriData> && std::is_same_v<TShape, Shape<CategoryTags::ThreeDArray>>) ||
      (IsMatrix<TOriData> && std::is_same_v<TShape, Shape<CategoryTags::Batch<CategoryTags::Matrix>>>) ||
      (IsMatrix<TOriData> && std::is_same_v<TShape, Shape<CategoryTags::Batch<CategoryTags::ThreeDArray>>>) ||
      (IsMatrix<TOriData> && std::is_same_v<TShape, Shape<CategoryTags::Sequence<CategoryTags::Matrix>>>) ||
      (IsMatrix<TOriData> && std::is_same_v<TShape, Shape<CategoryTags::Sequence<CategoryTags::ThreeDArray>>>) ||
      (IsMatrix<TOriData> && std::is_same_v<TShape, Shape<CategoryTags::BatchSequence<CategoryTags::Matrix>>>) ||
      (IsMatrix<TOriData> && std::is_same_v<TShape, Shape<CategoryTags::BatchSequence<CategoryTags::ThreeDArray>>>) ||

      (IsThreeDArray<TOriData> && std::is_same_v<TShape, Shape<CategoryTags::Batch<CategoryTags::ThreeDArray>>>) ||
      (IsThreeDArray<TOriData> && std::is_same_v<TShape, Shape<CategoryTags::Sequence<CategoryTags::ThreeDArray>>>) ||
      (IsThreeDArray<TOriData> && std::is_same_v<TShape, Shape<CategoryTags::BatchSequence<CategoryTags::ThreeDArray>>>) ||
      
      std::is_same_v<RemConstRef<decltype(std::declval<TOriData>().Shape())>, TShape>;
      
namespace OperDuplicate
{
template<typename TAimShape>
bool ShapeMatch(const Shape<CategoryTags::Scalar>&, const TAimShape&)
{
    return true;
};

template<typename TAimShape>
bool ShapeMatch(const Shape<CategoryTags::Matrix>& ori, const TAimShape& aim)
{
    return ((ori.RowNum() == aim.RowNum()) && (ori.ColNum() == aim.ColNum()));
};

template<typename TAimShape>
bool ShapeMatch(const Shape<CategoryTags::ThreeDArray>& ori, const TAimShape& aim)
{
    return ((ori.RowNum() == aim.RowNum()) &&
            (ori.ColNum() == aim.ColNum()) &&
            (ori.PageNum() == aim.PageNum()));
};

    template <typename TInputHandle, typename TShape, typename TOutputHandle>
    class EvalItem : public BaseEvalItem<DeviceTypeFromHandle<TOutputHandle>>
    {
        using BaseType = BaseEvalItem<DeviceTypeFromHandle<TOutputHandle>>;
    public:
        EvalItem(TInputHandle oriHandle, TShape shape, TOutputHandle outputHandle)
            : BaseType(std::type_index(typeid(EvalItem)),
                       {oriHandle.DataPtr()}, outputHandle.DataPtr())
            , m_inputHandle(std::move(oriHandle))
            , m_shape(std::move(shape))
            , m_outputHandle(std::move(outputHandle))
        {}
        
        const TInputHandle m_inputHandle;
        const TShape m_shape;
        TOutputHandle m_outputHandle;
    };

    template <typename TInputHandle, typename TShape, typename TOutputHandle>
    class EvalGroup : public TrivalEvalGroup<EvalItem<TInputHandle, TShape, TOutputHandle>>
    {
        using EvalItemType = EvalItem<TInputHandle, TShape, TOutputHandle>;
    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            using ResType = typename TOutputHandle::DataType;
            ResType out(evalItem.m_shape);

            const auto& in = evalItem.m_inputHandle.Data();
            using ElementType = typename ResType::ElementType;

            const size_t inCount = in.Shape().Count();
            auto low_in = LowerAccess(in);
            const ElementType* mem_in = low_in.RawMemory();

            const size_t outCount = out.Shape().Count();
            auto low_out = LowerAccess(out);
            ElementType* mem_out = low_out.MutableRawMemory();

            static_assert(std::is_same_v<typename EvalItemType::DeviceType, DeviceTags::CPU>,
                          "Currently only CPU is supported");

            assert(outCount % inCount == 0);
            const size_t loopCount = outCount / inCount;
            for (size_t i = 0; i < loopCount; ++i)
            {
                memcpy(mem_out, mem_in, sizeof(ElementType) * inCount);
                mem_out += inCount;
            }
            evalItem.m_outputHandle.SetData(std::move(out));
        }
    };

struct Calculator
{
    template <typename TEvalRes, typename TOriData, typename TShape>
    static void EvalRegister(TEvalRes& evalRes, const TOriData& oriData, const TShape& shape)
    {
        using DeviceType = typename TEvalRes::DataType::DeviceType;

        auto handle = oriData.EvalRegister();
        auto outHandle = evalRes.Handle();
        
        using ItemType = EvalItem<decltype(handle), TShape, decltype(outHandle)>;
        using GroupType = EvalGroup<decltype(handle), TShape, decltype(outHandle)>;
        using DispatcherType = TrivalEvalItemDispatcher<GroupType>;

        auto item = std::make_unique<ItemType>(std::move(handle), shape, std::move(outHandle));
        EvalPlan<DeviceType>::Inst().template Register<DispatcherType>(std::move(item));
    }
};
}

// Note: since operator<OpTags::Duplicate> cannot deduce its category from operands,
// so here involve a partial specification.
template <typename TOriData, typename TCategory>
class Operator<OpTags::Duplicate, TOriData, Shape<TCategory>>
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
            auto evalHandle = m_evalBuf.Handle();
            if (!EvalPlan<DeviceType>::Inst().IsAlreayRegisted(evalHandle.DataPtr()))
            {
                OperDuplicate::Calculator::EvalRegister(m_evalBuf, m_oriData, m_shape);
            }
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
          typename = std::enable_if_t<IsValidOper<OpTags::Duplicate, TOriData, RemConstRef<TShape>>>>
auto Duplicate(TOriData&& data, TShape&& shape)
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
        if (!OperDuplicate::ShapeMatch(data.Shape(), shape))
        {
            throw std::runtime_error("Cannot duplicate for un-match shape.");
        }
        using RawDataType = RemConstRef<TOriData>;
        using RawShapeType = RemConstRef<TShape>;
        
        using ResType = Operator<OpTags::Duplicate, RawDataType, RawShapeType>;
        return ResType(std::forward<TOriData>(data),
                       std::forward<TShape>(shape));
    }
}

template <typename TOriData, typename TShape>
auto DuplicateOrKeep(TOriData&& data, TShape&& shape)
{
    if constexpr (IsOutOfDataCategory<TOriData>)
    {
        return std::forward<TOriData>(data);
    }
    else
    {
        return Duplicate(std::forward<TOriData>(data), std::forward<TShape>(shape));
    }
}
}