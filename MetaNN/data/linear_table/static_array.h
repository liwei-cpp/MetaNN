#pragma once

#include <type_traits>
#include <MetaNN/data/facilities/continuous_memory.h>
#include <MetaNN/data/facilities/lower_access.h>
#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/facilities/traits.h>

namespace MetaNN
{
namespace NSStatciArray
{
    template <typename TShape, typename TP1, typename TP2, typename TP3, typename TP4, typename TP5, typename TVal>
    auto PosValSegment(const TShape& shape, TP1 p1, TP2 p2, TP3 p3, TP4 p4, TP5 p5, TVal val)
    {
        const size_t pos = shape.Index2Count(p1, p2, p3, p4, p5);
        return std::pair{pos, val};
    }

    template <typename TShape, typename TP1, typename TP2, typename TP3, typename TP4, typename TVal>
    auto PosValSegment(const TShape& shape, TP1 p1, TP2 p2, TP3 p3, TP4 p4, TVal val)
    {
        const size_t pos = shape.Index2Count(p1, p2, p3, p4);
        return std::pair{pos, val};
    }
    
    template <typename TShape, typename TP1, typename TP2, typename TP3, typename TVal>
    auto PosValSegment(const TShape& shape, TP1 p1, TP2 p2, TP3 p3, TVal val)
    {
        const size_t pos = shape.Index2Count(p1, p2, p3);
        return std::pair{pos, val};
    }
    
    template <typename TShape, typename TP1, typename TP2, typename TVal>
    auto PosValSegment(const TShape& shape, TP1 p1, TP2 p2, TVal val)
    {
        const size_t pos = shape.Index2Count(p1, p2);
        return std::pair{pos, val};
    }
    
    template <typename TShape, typename TP1, typename TVal>
    auto PosValSegment(const TShape& shape, TP1 p1, TVal val)
    {
        const size_t pos = shape.Index2Count(p1);
        return std::pair{pos, val};
    }
}

template <typename TElement, typename TDevice,
          template<typename>class TCateWrapper, typename TCardinalCate>
class StaticArray
{
    static_assert(std::is_same<RemConstRef<TElement>, TElement>::value,
                  "TElement is not an available type");
public:
    using CategoryTag = TCateWrapper<TCardinalCate>;
    using ElementType = TElement;
    using DeviceType = DeviceTags::CPU;
    
    friend struct LowerAccessImpl<StaticArray>;
public:
    explicit StaticArray(MetaNN::Shape<CategoryTag> p_shape = MetaNN::Shape<CategoryTag>())
        : m_shape(std::move(p_shape))
        , m_mem(m_shape.Count())
    {}
    
    template <typename... TShapeParams>
    explicit StaticArray(size_t val, TShapeParams&&... shapeParams)
        : m_shape(val, std::forward<TShapeParams>(shapeParams)...)
        , m_mem(m_shape.Count())
    {}
    
    template <typename TLensCont, typename... TShapeParams>
    explicit StaticArray(const TLensCont& p_lens,
                         TShapeParams&&... shapeParams)
        : m_shape(p_lens, std::forward<TShapeParams>(shapeParams)...)
        , m_mem(m_shape.Count())
    {}
    
    template <typename...TShapeParams>
    explicit StaticArray(ContinuousMemory<ElementType, DeviceType> p_mem,
                         TShapeParams&&... shapeParams)
        : m_shape(std::forward<TShapeParams>(shapeParams)...)
        , m_mem(std::move(p_mem))
    {}
    
    const auto& Shape() const noexcept
    {
        return m_shape;
    }
    
    bool operator== (const StaticArray& val) const
    {
        return (m_shape == val.m_shape) &&
               (m_mem == val.m_mem);
    }
    
    bool AvailableForWrite() const { return m_mem.IsShared(); }

    template <typename... TPosValParams>
    void SetValue(size_t Index1, TPosValParams... posValParams)
    {
        static_assert(std::is_same_v<DeviceType, DeviceTags::CPU>,
                      "Only CPU supports this method.");

        assert(AvailableForWrite());
        const auto [pos, val] = NSStatciArray::PosValSegment(m_shape, Index1, posValParams...);
        (m_mem.RawMemory())[pos] = val;
    }
    
    const auto operator [] (size_t id) const
    {
        if constexpr (IsBatchSequenceCategoryTag<CategoryTag>)
        {
            const auto& seqLenCont = m_shape.SeqLenContainer();
            if (id >= seqLenCont.size())
            {
                throw std::runtime_error("ID out of bound.");
            }
            
            size_t pos = std::accumulate(seqLenCont.begin(), seqLenCont.begin() + id, 0);
            
            const MetaNN::Shape<TCardinalCate>& cardinalShape = m_shape.Cardinal();
            pos *= cardinalShape.Count();
            
            using AimType = PrincipalDataType<CategoryTags::Sequence<TCardinalCate>, ElementType, DeviceType>;
            const MetaNN::Shape<CategoryTags::Sequence<TCardinalCate>> aimShape(seqLenCont[id], cardinalShape);
            return AimType(m_mem.Shift(pos), aimShape);
        }
        else
        {
            using AimType = PrincipalDataType<TCardinalCate, ElementType, DeviceType>;
            const MetaNN::Shape<TCardinalCate>& aimShape = m_shape.Cardinal();
            
            const size_t pos = id * aimShape.Count();
            if (pos >= m_shape.Count())
            {
                throw std::runtime_error("ID out of bound.");
            }
            return AimType(m_mem.Shift(pos), aimShape);
        }
    }
    
    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }
    
private:
    MetaNN::Shape<CategoryTag> m_shape;
    ContinuousMemory<ElementType, DeviceType> m_mem;
};

template <typename TElement, typename TDevice,
          template<typename>class TCateWrapper, typename TCardinalCate>
struct LowerAccessImpl<StaticArray<TElement, TDevice, TCateWrapper, TCardinalCate>>
{
    LowerAccessImpl(StaticArray<TElement, TDevice, TCateWrapper, TCardinalCate> p)
        : m_data(std::move(p))
    {}

    auto MutableRawMemory()
    {
        return m_data.m_mem.RawMemory();
    }

    const auto RawMemory() const
    {
        return m_data.m_mem.RawMemory();
    }

private:
    StaticArray<TElement, TDevice, TCateWrapper, TCardinalCate> m_data;
};
}