#pragma once
#include <MetaNN/data/facilities/continuous_memory.h>
#include <MetaNN/data/facilities/lower_access.h>
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
    
    friend struct LowerAccessImpl<Scalar>;

public:
    static Scalar CreateWithShape()
    {
        return Scalar{};
    }
    
public:
    explicit Scalar(MetaNN::Shape<CategoryTag> p_shape)
        : Scalar() {}

    Scalar(ElementType elem = ElementType())
        : m_mem(1)
    {
        SetValue(elem);
    }
                
    template <typename...TShapeParams>
    explicit Scalar(ContinuousMemory<ElementType, DeviceType> p_mem,
                    TShapeParams&&...)
        : m_mem(std::move(p_mem))
    {}
    
    const auto& Shape() const noexcept
    {
        const static MetaNN::Shape<CategoryTag> shape;
        return shape;
    }
    
    bool AvailableForWrite() const
    {
        return m_mem.IsShared();
    }
    
    void SetValue(ElementType val)
    {
        assert(AvailableForWrite());
        (m_mem.RawMemory())[0] = val;
    }
   
    auto Value() const noexcept
    {
        return (m_mem.RawMemory())[0];
    }
    
    bool operator== (const Scalar& val) const noexcept
    {
        return (Value() == val.Value());
    }

    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }

private:
    ContinuousMemory<ElementType, DeviceType> m_mem;
};

template<typename TElem, typename TDevice>
struct LowerAccessImpl<Scalar<TElem, TDevice>>
{
    LowerAccessImpl(Scalar<TElem, TDevice> p)
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
    Scalar<TElem, TDevice> m_data;
};
}