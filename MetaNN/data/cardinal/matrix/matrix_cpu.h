#pragma once

#include <MetaNN/data/cardinal/matrix/matrix_base.h>
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
class Matrix<TElem, DeviceTags::CPU> : public Shape<CategoryTags::Matrix>
{    
public:
    static_assert(std::is_same<RemConstRef<TElem>, TElem>::value,
                  "TElem is not an available type");
                  
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
    
    friend struct LowerAccessImpl<Matrix<TElem, DeviceTags::CPU>>;
    
public:
    Matrix(size_t p_rowNum = 0, size_t p_colNum = 0)
        : Shape<CategoryTags::Matrix>(p_rowNum, p_colNum)
        , m_mem(p_rowNum * p_colNum)
    {}
    
    Matrix(ContinuousMemory<ElementType, DeviceType> p_mem,
           size_t p_rowNum,
           size_t p_colNum)
        : Shape<CategoryTags::Matrix>(p_rowNum, p_colNum)
        , m_mem(std::move(p_mem))
    {}
    
    bool operator== (const Matrix& val) const
    {
        return (GetShape() == val.GetShape()) &&
               (m_mem == val.m_mem);
    }

    template <typename TOtherType,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<TOtherType>, Matrix>>>
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

    void SetValue(size_t p_rowId, size_t p_colId, ElementType val)
    {
        assert(AvailableForWrite());
        assert((p_rowId < RowNum()) && (p_colId < ColNum()));
        (m_mem.RawMemory())[p_rowId * ColNum() + p_colId] = val;
    }

    const auto operator () (size_t p_rowId, size_t p_colId) const
    {
        assert((p_rowId < RowNum()) && (p_colId < ColNum()));
        return (m_mem.RawMemory())[p_rowId * ColNum() + p_colId];
    }

    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }
    
private:
    ContinuousMemory<ElementType, DeviceType> m_mem;
};

template<typename TElem>
struct LowerAccessImpl<Matrix<TElem, DeviceTags::CPU>>
{
    LowerAccessImpl(Matrix<TElem, DeviceTags::CPU> p)
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
    Matrix<TElem, DeviceTags::CPU> m_matrix;
};
}
