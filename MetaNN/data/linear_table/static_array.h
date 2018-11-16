#pragma once

#include <type_traits>
#include <MetaNN/data/facilities/continuous_memory.h>
#include <MetaNN/data/facilities/lower_access.h>
#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/facilities/traits.h>

namespace MetaNN
{
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
    template<typename...TShapeParams>
    static StaticArray Create(TShapeParams&&... shapeParams)
    {
        MetaNN::Shape<CategoryTag> shape(std::forward<TShapeParams>(shapeParams)...);
        return StaticArray(std::move(shape));
    }

public:
    explicit StaticArray(MetaNN::Shape<CategoryTag> p_shape = MetaNN::Shape<CategoryTag>())
        : m_shape(std::move(p_shape))
        , m_mem(m_shape.Count())
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

    template <typename... TPosParams>
    void SetValue(ElementType val, TPosParams... posParams)
    {
        static_assert(std::is_same_v<DeviceType, DeviceTags::CPU>,
                      "Only CPU supports this method.");
                      
        assert(AvailableForWrite());
        const size_t pos = m_shape.Index2Count(posParams...);
        (m_mem.RawMemory())[pos] = val;
    }
    
    const auto operator [] (size_t id) const
    {
        if constexpr (IsBatchSequenceategoryTag<CategoryTag>)
        {
            const auto& seqLenCont = m_shape.SeqLenContainer();
            if (id >= seqLenCont.size())
            {
                throw std::runtime_error("ID out of bound.");
            }
            
            size_t pos = std::accumulate(seqLenCont.begin(), seqLenCont.begin() + id, 0);
            
            const MetaNN::Shape<TCardinalCate>& cardinalShape = static_cast<const MetaNN::Shape<TCardinalCate>&>(m_shape);
            pos *= cardinalShape.Count();
            
            using AimType = PrincipalDataType<CategoryTags::Sequence<CategoryTag>, ElementType, DeviceType>;
            const MetaNN::Shape<CategoryTags::Sequence<TCardinalCate>> aimShape(seqLenCont[id], cardinalShape);
            return AimType(m_mem.Shift(pos), aimShape);
        }
        else
        {
            using AimType = PrincipalDataType<TCardinalCate, ElementType, DeviceType>;
            const MetaNN::Shape<TCardinalCate>& aimShape = static_cast<const MetaNN::Shape<TCardinalCate>&>(m_shape);
            
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