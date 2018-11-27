#pragma once
#include <MetaNN/data/facilities/continuous_memory.h>
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
    static Scalar CreateWithShape()
    {
        return Scalar{};
    }
    
public:
    Scalar(ElementType elem = ElementType())
        : m_elem(elem) {}
        
    template <typename...TShapeParams>
    explicit Scalar(ContinuousMemory<ElementType, DeviceType> p_mem,
                    TShapeParams&&... shapeParams)
        : m_elem((p_mem.RawMemory())[0])
    {}
    
    const auto& Shape() const noexcept
    {
        const static MetaNN::Shape<CategoryTag> shape;
        return shape;
    }
    
    void SetValue(ElementType val)
    {
        m_elem = val;
    }
   
    auto Value() const noexcept
    {
        return m_elem;
    }
    
    bool operator== (const Scalar& val) const noexcept
    {
        return (m_elem == val.m_elem);
    }

    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }

private:
    ElementType m_elem;
};
}