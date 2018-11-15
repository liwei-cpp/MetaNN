#pragma once

#include <MetaNN/data/facilities/continuous_memory.h>
#include <MetaNN/data/facilities/lower_access.h>
#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>

namespace MetaNN
{
template<typename TElem, typename TDevice>
class Matrix
{
    static_assert(std::is_same<RemConstRef<TElem>, TElem>::value,
                  "TElem is not an available type");
public:
    using CategoryTag = CategoryTags::Matrix;
    using ElementType = TElem;
    using DeviceType = TDevice;
    
    friend struct LowerAccessImpl<Matrix>;

public:
    explicit Matrix(size_t rowNum, size_t colNum)
        : m_shape(rowNum, colNum)
        , m_mem(m_shape.Count())
    {}
    
    explicit Matrix(MetaNN::Shape<CategoryTag> p_shape = MetaNN::Shape<CategoryTag>())
        : m_shape(std::move(p_shape))
        , m_mem(m_shape.Count())
    {}
    
    template <typename...TShapeParams>
    explicit Matrix(ContinuousMemory<ElementType, DeviceType> p_mem,
           TShapeParams&&... shapeParams)
        : m_shape(std::forward<TShapeParams>(shapeParams)...)
        , m_mem(std::move(p_mem))
    {}
    
    const auto& Shape() const noexcept
    {
        return m_shape;
    }
    
    bool operator== (const Matrix& val) const
    {
        return (Shape() == val.Shape()) &&
               (m_mem == val.m_mem);
    }

    bool AvailableForWrite() const
    {
        return m_mem.IsShared();
    }

    void SetValue(size_t p_rowId, size_t p_colId, ElementType val)
    {
        static_assert(std::is_same_v<DeviceType, DeviceTags::CPU>,
                      "Only CPU supports this method.");
                      
        assert(AvailableForWrite());
        const size_t pos = m_shape.Index2Count(p_rowId, p_colId);
        (m_mem.RawMemory())[pos] = val;
    }

    const auto operator () (size_t p_rowId, size_t p_colId) const
    {
        static_assert(std::is_same_v<DeviceType, DeviceTags::CPU>,
                      "Only CPU supports this method.");

        const size_t pos = m_shape.Index2Count(p_rowId, p_colId);
        return (m_mem.RawMemory())[pos];
    }

    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }
    
protected:
    MetaNN::Shape<CategoryTag> m_shape;
    ContinuousMemory<ElementType, DeviceType> m_mem;
};

template<typename TElem, typename TDevice>
struct LowerAccessImpl<Matrix<TElem, TDevice>>
{
    LowerAccessImpl(Matrix<TElem, TDevice> p)
        : m_matrix(std::move(p))
    {}

    auto MutableRawMemory()
    {
        return m_matrix.m_mem.RawMemory();
    }

    const auto RawMemory() const
    {
        return m_matrix.m_mem.RawMemory();
    }

private:
    Matrix<TElem, TDevice> m_matrix;
};
}