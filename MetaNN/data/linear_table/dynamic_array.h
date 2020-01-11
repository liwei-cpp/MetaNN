#pragma once

#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/data/facilities/category_tags.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/facilities/traits.h>
#include <MetaNN/evaluate/eval_buffer.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <memory>

namespace MetaNN
{
namespace NSDynamicArray
{
template <typename TOriCate, template<typename> class CateWrapper>
struct Helper_;

template <typename TOriCate>
struct Helper_<TOriCate, CategoryTags::Sequence>
{
    using CardinalType = TOriCate;
    using PromoteType = CategoryTags::Sequence<CardinalType>;
    
    static const Shape<PromoteType> ShapeInit(const Shape<CardinalType>& cardShape)
    {
        return Shape<PromoteType>(0, cardShape);
    }
    
    static bool TryUpdateShape(Shape<PromoteType>& proShape, const Shape<CardinalType>& cardShape)
    {
        if (proShape.Cardinal() != cardShape)
            return false;
        ++proShape.Length();
        return true;
    }
};

template <typename TOriCate>
struct Helper_<TOriCate, CategoryTags::Batch>
{
    using CardinalType = TOriCate;
    using PromoteType = CategoryTags::Batch<CardinalType>;
    
    static const Shape<PromoteType> ShapeInit(const Shape<CardinalType>& cardShape)
    {
        return Shape<PromoteType>(0, cardShape);
    }
    
    static bool TryUpdateShape(Shape<PromoteType>& proShape, const Shape<CardinalType>& cardShape)
    {
        if (proShape.Cardinal() != cardShape)
            return false;
        ++proShape.BatchNum();
        return true;
    }
};

template <typename TOriCate>
struct Helper_<CategoryTags::Sequence<TOriCate>, CategoryTags::Batch>
{
    using CardinalType = TOriCate;
    using PromoteType = CategoryTags::BatchSequence<CardinalType>;
    
    static const Shape<PromoteType> ShapeInit(const Shape<CardinalType>& cardShape)
    {
        return Shape<PromoteType>({}, cardShape);
    }
    
    static const Shape<PromoteType> ShapeInit(const Shape<CategoryTags::Sequence<CardinalType>>& oriShape)
    {
        return Shape<PromoteType>({}, oriShape.Cardinal());
    }
    
    static bool TryUpdateShape(Shape<PromoteType>& proShape, const Shape<CategoryTags::Sequence<CardinalType>>& oriShape)
    {
        if (proShape.Cardinal() != oriShape.Cardinal())
            return false;
        proShape.SeqLenContainer().push_back(oriShape.Length());
        return true;
    }
};

    template <typename TInputHandle, typename TElement, typename TDevice, typename TCategory>
    class EvalItem : public BaseEvalItem<TDevice>
    {
    public:
        using PrincipleType = PrincipalDataType<TCategory, TElement, TDevice>;
        using TOutputHandle = EvalHandle<PrincipleType>;
        EvalItem(std::vector<TInputHandle> p_input, TOutputHandle p_output,
                 MetaNN::Shape<TCategory> p_outputShape,
                 std::vector<const void*> p_dependencies)
            : BaseEvalItem<TDevice>(std::type_index(typeid(EvalItem)),
                                    std::move(p_dependencies), p_output.DataPtr())
            , m_inputs(std::move(p_input))
            , m_output(std::move(p_output))
            , m_outputShape(std::move(p_outputShape))
        {}

        std::vector<TInputHandle> m_inputs;
        TOutputHandle m_output;
        MetaNN::Shape<TCategory> m_outputShape;
    };
    
    template <typename TInputHandle, typename TElement, typename TDevice, typename TCategory>
    class EvalGroup : public TrivalEvalGroup<EvalItem<TInputHandle, TElement, TDevice, TCategory>>
    {
        using EvalItemType = EvalItem<TInputHandle, TElement, TDevice, TCategory>;
    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            using ResType = typename EvalItemType::TOutputHandle::DataType;
            ResType res(evalItem.m_outputShape);
            using TElem = typename ResType::ElementType;
        
            static_assert(std::is_same_v<TDevice, DeviceTags::CPU>,
                          "Currently only CPU is supported");

            auto lowerRes = LowerAccess(res);
            TElem* resMem = lowerRes.MutableRawMemory();
        
            size_t startPoint = 0;
            for (size_t i = 0; i < evalItem.m_inputs.size(); ++i)
            {
                const auto& curItem = evalItem.m_inputs[i].Data();
                if constexpr (IsScalar<decltype(curItem)>)
                {
                    *resMem = curItem.Value();
                }
                else
                {
                    auto lowerItem = LowerAccess(curItem);
                    const TElem* itemMem = lowerItem.RawMemory();
                    memcpy(resMem, itemMem, sizeof(TElem) * curItem.Shape().Count());                
                }
                startPoint += curItem.Shape().Count();
                resMem += curItem.Shape().Count();
            }
        
            assert(startPoint == res.Shape().Count());
            evalItem.m_output.SetData(std::move(res));
        }
    };
}

template <typename TData, template<typename> class CateWrapper>
class DynamicArray
{
    static_assert(std::is_same_v<RemConstRef<TData>, TData>);
    
    using HelperBlob = NSDynamicArray::Helper_<typename TData::CategoryTag, CateWrapper>;
    using CardinalTag = typename HelperBlob::CardinalType;
    
public:
    using CategoryTag = typename HelperBlob::PromoteType;
    using ElementType = typename TData::ElementType;
    using DeviceType = typename TData::DeviceType;

private:
    using PrincipleType = PrincipalDataType<CategoryTag, ElementType, DeviceType>;

public:
    explicit DynamicArray(MetaNN::Shape<CardinalTag> p_shape = MetaNN::Shape<CardinalTag>())
        : m_shape(HelperBlob::ShapeInit(std::move(p_shape)))
    {}
    
    template <typename... TShapeParams>
    explicit DynamicArray(size_t val, TShapeParams&&... shapeParams)
        : DynamicArray(MetaNN::Shape<CardinalTag>(val, std::forward<TShapeParams>(shapeParams)...))
    {}
    
    template <typename TIterator,
              typename = std::enable_if_t<IsIterator<TIterator>>>
    DynamicArray(TIterator b, TIterator e)
        : m_shape(HelperBlob::ShapeInit(b->Shape()))
    {
        if (b == e)
        {
            throw std::runtime_error("Cannot initialize dynamic array with empty sequence.");
        }
        
        for (auto cur = b; cur != e; ++cur)
        {
            PushBack(*cur);
        }
    }
    
    const auto& Shape() const noexcept
    {
        return m_shape;
    }
    
    bool AvailableForWrite() const noexcept
    {
        return (!m_evalBuf.IsEvaluated()) && (m_buffer.use_count() == 1);
    }

    void PushBack(TData data)
    {
        if (m_buffer->empty() && (m_shape.Cardinal() == MetaNN::Shape<CardinalTag>()))
        {
            m_shape = HelperBlob::ShapeInit(data.Shape());
        }

        if (!HelperBlob::TryUpdateShape(m_shape, data.Shape()))
        {
            throw std::runtime_error("Shape mismatch");
        }
        m_buffer->push_back(std::move(data));
    }
    
    template <typename...TArgs>
    void EmplaceBack(TArgs&&... args)
    {
        assert(AvailableForWrite());
        TData tmp(std::forward<TArgs>(args)...);
        PushBack(std::move(tmp));
    }
    
    void Reserve(size_t num)
    {
        assert(AvailableForWrite());
        m_buffer.reserve(num);
    }
    
    void Clear()
    {
        assert(AvailableForWrite());
        m_buffer.clear();
        m_shape = HelperBlob::ShapeInit(m_shape.Cardinal());
    }
    
    void Reverse()
    {
        assert(AvailableForWrite());
        if (!m_buffer) return;
        auto& cont = *m_buffer;
        std::reverse(cont.begin(), cont.end());
    }
    
    bool IsEmpty() const
    {
        return m_buffer->empty();
    }
    
    const auto& operator[] (size_t id) const
    {
        return (*m_buffer)[id];
    }
    
    bool operator== (const DynamicArray& val) const
    {
        return m_buffer == val.m_buffer;
    }

    auto EvalRegister() const
    {
        using TOpEvalHandle = std::decay_t<decltype(std::declval<TData>().EvalRegister())>;
        using TEvalItem = NSDynamicArray::EvalItem<TOpEvalHandle, ElementType, DeviceType, CategoryTag>;
        using TEvalGroup = NSDynamicArray::EvalGroup<TOpEvalHandle, ElementType, DeviceType, CategoryTag>;
        using TItemDispatcher = TrivalEvalItemDispatcher<TEvalGroup>;

        if (!m_evalBuf.IsEvaluated())
        {
            auto outHandle = m_evalBuf.Handle();
            if (!EvalPlan<DeviceType>::Inst().IsAlreayRegisted(outHandle.DataPtr()))
            {
                std::vector<TOpEvalHandle> handleBuf;
                std::vector<const void*> depVec;
                handleBuf.reserve(m_buffer->size());
                depVec.reserve(m_buffer->size());
                for (size_t i = 0; i < m_buffer->size(); ++i)
                {
                    handleBuf.push_back((*m_buffer)[i].EvalRegister());
                    depVec.push_back(handleBuf.back().DataPtr());
                }
                EvalPlan<DeviceType>::Inst().template Register<TItemDispatcher>(
                        std::make_unique<TEvalItem>(std::move(handleBuf), std::move(outHandle),
                                                    m_shape, std::move(depVec)));
            }
        }
        return m_evalBuf.ConstHandle();
    }
    
private:
    MetaNN::Shape<CategoryTag> m_shape;
    std::shared_ptr<std::vector<TData>> m_buffer = std::make_shared<std::vector<TData>>();
    EvalBuffer<PrincipleType> m_evalBuf;
};
}
