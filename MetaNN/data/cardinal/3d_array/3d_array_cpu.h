#pragma once

#include <MetaNN/data/cardinal/3d_array/3d_array_base.h>
#include <MetaNN/data/facilities/continuous_memory.h>
#include <MetaNN/data/facilities/lower_access.h>
#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>
#include <cassert>
#include <type_traits>

namespace MetaNN
{
template <typename TElem>
class ThreeDArray<TElem, DeviceTags::CPU> : public Shape_<CategoryTags::ThreeDArray>
{    
public:
    static_assert(std::is_same<RemConstRef<TElem>, TElem>::value,
                  "TElem is not an available type");
                  
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
    
    friend struct LowerAccessImpl<ThreeDArray<TElem, DeviceTags::CPU>>;

public:
    ThreeDArray(size_t p_pageNum = 0, size_t p_rowNum = 0, size_t p_colNum = 0)
        : Shape_<CategoryTags::ThreeDArray>(p_pageNum, p_rowNum, p_colNum)
        , m_mem(p_pageNum * p_rowNum * p_colNum)
    {}
    
    ThreeDArray(ContinuousMemory<ElementType, DeviceType> p_mem,
                size_t p_pageNum,
                size_t p_rowNum,
                size_t p_colNum)
        : Shape_<CategoryTags::ThreeDArray>(p_pageNum, p_rowNum, p_colNum)
        , m_mem(std::move(p_mem))
    {}


    bool operator== (const ThreeDArray& val) const
    {
        return (Shape() == val.Shape()) &&
               (m_mem == val.m_mem);
    }

    template <typename TOtherType,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<TOtherType>, ThreeDArray>>>
    bool operator== (const TOtherType&) const
    {
        return false;
    }

    template <typename TData>
    bool operator!= (const TData& val) const
    {
        return !(operator==(val));
    }

    bool AvailableForWrite() const { return m_mem.IsShared(); }

    void SetValue(size_t p_pageId, size_t p_rowId, size_t p_colId, ElementType val)
    {
        assert(AvailableForWrite());
        assert((p_pageId < PageNum()) && (p_rowId < RowNum()) && (p_colId < ColNum()));
        
        (m_mem.RawMemory())[(p_pageId * RowNum() + p_rowId) * ColNum() + p_colId] = val;
    }

    const auto operator () (size_t p_pageId, size_t p_rowId, size_t p_colId) const
    {
        assert((p_pageId < PageNum()) && (p_rowId < RowNum()) && (p_colId < ColNum()));
        return (m_mem.RawMemory())[(p_pageId * RowNum() + p_rowId) * ColNum() + p_colId];
    }

    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }

private:
    ContinuousMemory<ElementType, DeviceType> m_mem;
};

template<typename TElem>
struct LowerAccessImpl<ThreeDArray<TElem, DeviceTags::CPU>>
{
    LowerAccessImpl(ThreeDArray<TElem, DeviceTags::CPU> p)
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
    ThreeDArray<TElem, DeviceTags::CPU> m_data;
};
}