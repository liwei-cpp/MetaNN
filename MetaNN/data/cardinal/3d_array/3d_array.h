#pragma once

#include <cassert>
#include <type_traits>
#include <MetaNN/data/facilities/continuous_memory.h>
#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/data/facilities/category_tags.h>
#include <MetaNN/data/facilities/lower_access.h>
#include <MetaNN/facilities/traits.h>
namespace MetaNN
{
template<typename TElem, typename TDevice>
class ThreeDArray
{
    static_assert(std::is_same_v<RemConstRef<TElem>, TElem>,
                  "TElem is not an available type");
public:
    using CategoryTag = CategoryTags::ThreeDArray;
    using ElementType = TElem;
    using DeviceType = TDevice;
    
    friend struct LowerAccessImpl<ThreeDArray>;
    
public:
    explicit ThreeDArray(MetaNN::Shape<CategoryTag> p_shape = MetaNN::Shape<CategoryTag>())
        : m_shape(std::move(p_shape))
        , m_mem(m_shape.Count())
    {}
    
    template <typename...TShapeParams>
    explicit ThreeDArray(size_t val, TShapeParams&&... shapeParams)
        : m_shape(val, std::forward<TShapeParams>(shapeParams)...)
        , m_mem(m_shape.Count())
    {}
    
    template <typename...TShapeParams>
    explicit ThreeDArray(ContinuousMemory<ElementType, DeviceType> p_mem,
                         TShapeParams&&... shapeParams)
        : m_shape(std::forward<TShapeParams>(shapeParams)...)
        , m_mem(std::move(p_mem))
    {}
    
    const auto& Shape() const noexcept
    {
        return m_shape;
    }
    
    bool operator== (const ThreeDArray& val) const
    {
        return (m_shape == val.m_shape) &&
               (m_mem == val.m_mem);
    }
    
    bool AvailableForWrite() const
    {
        return m_mem.IsShared();
    }

    void SetValue(size_t p_pageId, size_t p_rowId, size_t p_colId, ElementType val)
    {
        static_assert(std::is_same_v<DeviceType, DeviceTags::CPU>,
                      "Only CPU supports this method.");
        assert(AvailableForWrite());
        const size_t pos = m_shape.Index2Count(p_pageId, p_rowId, p_colId);
        (m_mem.RawMemory())[pos] = val;
    }

    const auto operator () (size_t p_pageId, size_t p_rowId, size_t p_colId) const
    {
        static_assert(std::is_same_v<DeviceType, DeviceTags::CPU>,
                      "Only CPU supports this method.");

        const size_t pos = m_shape.Index2Count(p_pageId, p_rowId, p_colId);
        return (m_mem.RawMemory())[pos];
    }

    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }
private:
    MetaNN::Shape<CategoryTag> m_shape;
    ContinuousMemory<ElementType, DeviceType> m_mem;
};

template<typename TElem, typename TDevice>
struct LowerAccessImpl<ThreeDArray<TElem, TDevice>>
{
    LowerAccessImpl(ThreeDArray<TElem, TDevice> p)
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
    ThreeDArray<TElem, TDevice> m_data;
};
}