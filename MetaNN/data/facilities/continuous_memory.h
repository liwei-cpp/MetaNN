#pragma once

#include <MetaNN/data/facilities/allocators.h>
#include <MetaNN/facilities/traits.h>
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
    {}

    ContinuousMemory Shift(size_t pos) const
    {
        return ContinuousMemory(m_mem, pos);
    }
    
    auto RawMemory() const
    {
        return m_mem.get();
    }

    bool IsShared() const
    {
        return m_mem.use_count() == 1;
    }
    
    bool operator== (const ContinuousMemory& val) const noexcept
    {
        return (m_mem == val.m_mem);
    }

    bool operator!= (const ContinuousMemory& val) const noexcept
    {
        return !(operator==(val));
    }

private:
    ContinuousMemory(const std::shared_ptr<ElementType>& owner, size_t p_shift)
        : m_mem(owner, owner.get() + p_shift)
    {}
    
private:
    std::shared_ptr<ElementType> m_mem;
};
}