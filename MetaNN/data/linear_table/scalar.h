#pragma once

#include <MetaNN/data/linear_table/linear_table.h>
#include <MetaNN/data/facilities/continuous_memory.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>
#include <cassert>

namespace MetaNN
{
template <typename TElem>
class LinearTable<TElem, DeviceTags::CPU, CategoryTags::Scalar>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
    
    friend LowerAccessImpl<LinearTable<TElem, DeviceTags::CPU, CategoryTags::Scalar>>;
    
public:
    LinearTable(size_t length = 0)
        : m_mem(length)
        , m_len(length) {}

    bool AvailableForWrite() const { return m_mem.UseCount() == 1; }

    void SetValue(size_t p_id, ElementType val)
    {
        assert(AvailableForWrite());
        assert(p_id < m_len);
        (m_mem.RawMemory())[p_id] = val;
    }
    
    const auto operator[](size_t p_id) const
    {
        assert(p_id < m_len);
        return (m_mem.RawMemory())[p_id];
    }
   
    bool operator== (const LinearTable& val) const
    {
        return (m_mem == val.m_mem) && (m_len == val.m_len);
    }

    template <typename TOtherType>
    bool operator== (const TOtherType&) const
    {
        return false;
    }

    template <typename TData>
    bool operator!= (const TData& val) const
    {
        return !(operator==(val));
    }
    
protected:
    size_t Count() const { return m_len; }
private:
    ContinuousMemory<ElementType, DeviceType> m_mem;
    size_t m_len;
};

template<typename TElem>
struct LowerAccessImpl<LinearTable<TElem, DeviceTags::CPU, CategoryTags::Scalar>>
{
    LowerAccessImpl(LinearTable<TElem, DeviceTags::CPU, CategoryTags::Scalar> p)
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
    LinearTable<TElem, DeviceTags::CPU, CategoryTags::Scalar> m_data;
};
}