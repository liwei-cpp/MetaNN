#pragma once

#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>

namespace MetaNN
{
template <typename TElem, typename TDevice = DeviceTags::CPU>
class Scalar;

template <typename TElem>
class Scalar<TElem, DeviceTags::CPU>
{
    static_assert(std::is_same<RemConstRef<TElem>, TElem>::value);
    
public:
    using CategoryTag = CategoryTags::Scalar;
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
    
public:
    Scalar(ElementType elem = ElementType())
        : m_shape()
        , m_elem(elem) {}
    
    const auto& Shape() const noexcept
    {
        return m_shape;
    }
    
    auto& Value() noexcept
    {
        return m_elem;
    }
   
    auto Value() const noexcept
    {
        return m_elem;
    }
    
    bool operator== (const Scalar& val) const noexcept
    {
        return (Shape() == val.Shape()) &&
               (m_elem == val.m_elem);
    }

    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }

private:
    MetaNN::Shape<CategoryTag> m_shape;
    ElementType m_elem;
};
}