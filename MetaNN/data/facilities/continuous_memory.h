#pragma once

#include <MetaNN/data/facilities/allocators.h>
#include <MetaNN/facilities/traits.h>
#include <cassert>

namespace MetaNN
{
template <typename TElem, typename TDevice>
class ContinuousMemory
{
    static_assert(std::is_same<RemConstRef<TElem>, TElem>::value);
    using ElementType = TElem;
    
public:
    explicit ContinuousMemory(size_t p_size)
        : m_mem(Allocator<TDevice>::template Allocate<ElementType>(p_size))
        , m_size(p_size)
    {}

    ContinuousMemory Shift(size_t pos) const
    {
        assert(pos < m_size);
        return ContinuousMemory(std::shared_ptr<ElementType>(m_mem, m_mem.get() + pos),
                                m_size - pos);
    }
    
    auto RawMemory() const
    {
        return m_mem.get();
    }

    bool IsShared() const
    {
        return m_mem.use_count() > 1;
    }
    
    size_t Size() const
    {
        return m_size;
    }
    
    bool operator== (const ContinuousMemory& val) const noexcept
    {
        return (m_mem == val.m_mem) && (m_size == val.m_size);
    }

    bool operator!= (const ContinuousMemory& val) const noexcept
    {
        return !(operator==(val));
    }

private:
    ContinuousMemory(std::shared_ptr<ElementType> ptr, size_t p_size)
        : m_mem(std::move(ptr))
        , m_size(p_size)
    {}
    
private:
    std::shared_ptr<ElementType> m_mem;
    size_t m_size;
};
}