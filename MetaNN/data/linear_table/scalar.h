#pragma once

#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/data/facilities/continuous_memory.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/data/facilities/lower_access.h>
#include <MetaNN/data/linear_table/linear_table_base.h>
#include <cassert>

namespace MetaNN
{
template <typename TElem, template <typename> class TCateWrapper>
class LinearTable<TElem, DeviceTags::CPU, TCateWrapper<CategoryTags::Scalar>>
    : public Shape_<TCateWrapper<CategoryTags::Scalar>>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
    
    friend LowerAccessImpl<LinearTable<TElem, DeviceTags::CPU, CategoryTags::Scalar>>;
    
public:
    LinearTable(size_t length = 0)
        : Shape_<TCateWrapper<CategoryTags::Scalar>>(length)
        , m_mem(length) {}

    bool AvailableForWrite() const { return m_mem.IsShared(); }

    void SetValue(size_t p_id, ElementType val)
    {
        using TShape = Shape_<TCateWrapper<CategoryTags::Scalar>>;
        assert(AvailableForWrite());
        assert(p_id < NSLinearTable::WrapperDim(TShape::Shape()));
        (m_mem.RawMemory())[p_id] = val;
    }
    
    const auto operator[](size_t p_id) const
    {
        using TShape = Shape_<TCateWrapper<CategoryTags::Scalar>>;
        assert(p_id < NSLinearTable::WrapperDim(TShape::Shape()));
        return (m_mem.RawMemory())[p_id];
    }
   
    bool operator== (const LinearTable& val) const
    {
        using TShape = Shape_<TCateWrapper<CategoryTags::Scalar>>;
        return (TShape::Shape() == val.Shape()) &&
               (m_mem == val.m_mem);
    }

    template <typename TOtherType,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<TOtherType>, LinearTable>>>
    bool operator== (const TOtherType&) const
    {
        return false;
    }

    template <typename TData>
    bool operator!= (const TData& val) const
    {
        return !(operator==(val));
    }
    
    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }

private:
    ContinuousMemory<ElementType, DeviceType> m_mem;
};

template<typename TElem, template <typename> class TCateWrapper>
struct LowerAccessImpl<LinearTable<TElem, DeviceTags::CPU, TCateWrapper<CategoryTags::Scalar>>>
{
    LowerAccessImpl(LinearTable<TElem, DeviceTags::CPU, TCateWrapper<CategoryTags::Scalar>> p)
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
    LinearTable<TElem, DeviceTags::CPU, TCateWrapper<CategoryTags::Scalar>> m_data;
};
}