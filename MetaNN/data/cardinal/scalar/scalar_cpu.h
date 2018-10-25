#pragma once
#include <MetaNN/data/cardinal/scalar/scalar_base.h>
#include <MetaNN/data/facilities/shape.h>

namespace MetaNN
{
template <typename TElem>
class Scalar<TElem, DeviceTags::CPU> : public Shape<CategoryTags::Scalar>
{
    static_assert(std::is_same<RemConstRef<TElem>, TElem>::value);
    
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
    
private:
    using ShapeType = Shape<CategoryTags::Scalar>;
public:
    Scalar(ElementType elem = ElementType())
        : m_elem(elem) {}
     
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
        return (ShapeType::operator==(val)) &&
               (m_elem == val.m_elem);
    }

    template <typename TOtherType,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<TOtherType>, Scalar>>>
    bool operator== (const TOtherType&) const noexcept
    {
        return false;
    }

    template <typename TData>
    bool operator!= (const TData& val) const noexcept
    {
        return !(operator==(val));
    }
    
    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }
    
    const auto& Shape() const noexcept
    {
        return static_cast<const ShapeType&>(*this);
    }
    
private:
    ElementType m_elem;
};
}