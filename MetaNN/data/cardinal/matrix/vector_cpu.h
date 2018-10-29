#pragma once

#include <MetaNN/data/cardinal/matrix/vector_base.h>
#include <MetaNN/data/facilities/continuous_memory.h>
#include <MetaNN/data/facilities/lower_access.h>
#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>
#include <cassert>
#include <cstring>
#include <type_traits>

namespace MetaNN
{
template <typename TElem>
class Vector<TElem, DeviceTags::CPU> : public Shape_<CategoryTags::Matrix>
{    
public:
    static_assert(std::is_same<RemConstRef<TElem>, TElem>::value,
                  "TElem is not an available type");
                  
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
    
    friend struct LowerAccessImpl<Vector<TElem, DeviceTags::CPU>>;

public:
    explicit Vector(size_t size = 0)
        : Shape_<CategoryTags::Matrix>(1, size)
        , m_mem(size)
    {}
    
    bool operator== (const Vector& val) const
    {
        return (Shape() == val.Shape()) &&
               (m_mem == val.m_mem);
    }

    template <typename TOtherType,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<TOtherType>, Vector>>>
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

    void SetValue(size_t p_id, ElementType val)
    {
        assert(AvailableForWrite());
        assert(p_id < ColNum());
        (m_mem.RawMemory())[p_id] = val;
    }

    const auto operator () (size_t p_id) const
    {
        assert(p_id < ColNum());
        return (m_mem.RawMemory())[p_id];
    }

    auto EvalRegister() const
    {
        return MakeConstEvalHandle(Matrix<ElementType, DeviceType>(m_mem, 1, ColNum()));
    }
    
private:
    ContinuousMemory<ElementType, DeviceType> m_mem;
};

template<typename TElem>
struct LowerAccessImpl<Vector<TElem, DeviceTags::CPU>>
{
    LowerAccessImpl(Vector<TElem, DeviceTags::CPU> p)
        : m_data(std::move(p))
    {}

    const auto RawMemory() const
    {
        return m_data.m_mem.RawMemory();
    }

private:
    Vector<TElem, DeviceTags::CPU> m_data;
};
}
