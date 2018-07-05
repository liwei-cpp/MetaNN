#pragma once

#include <MetaNN/data/facilities/allocators.h>
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
        , m_memStart(m_mem.get())
    {}

    ContinuousMemory(std::shared_ptr<ElementType> p_mem, ElementType* p_memStart)
        : m_mem(std::move(p_mem))
        , m_memStart(p_memStart)
    {}

    auto RawMemory() const { return m_memStart; }

    const std::shared_ptr<ElementType> SharedPtr() const
    {
        return m_mem;
    }

    size_t UseCount() const
    {
        return m_mem.use_count();
    }

    bool operator== (const ContinuousMemory& val) const
    {
        return (m_mem == val.m_mem) && (m_memStart == val.m_memStart);
    }

    bool operator!= (const ContinuousMemory& val) const
    {
        return !(operator==(val));
    }

private:
    std::shared_ptr<ElementType> m_mem;
    ElementType*                 m_memStart;
};
}