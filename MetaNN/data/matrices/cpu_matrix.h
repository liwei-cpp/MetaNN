#pragma once

#include <MetaNN/data/facilities/continuous_memory.h>
#include <MetaNN/data/facilities/lower_access.h>
#include <MetaNN/data/matrices/matrices.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>
#include <cassert>
#include <cstring>
#include <type_traits>

namespace MetaNN
{
template <typename TElem>
class Matrix<TElem, DeviceTags::CPU>
{    
public:
    static_assert(std::is_same<RemConstRef<TElem>, TElem>::value,
                  "TElem is not an available type");
                  
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
    
    friend struct LowerAccessImpl<Matrix<TElem, DeviceTags::CPU>>;

public:
    Matrix(size_t p_rowNum = 0, size_t p_colNum = 0)
        : m_mem(p_rowNum * p_colNum)
        , m_rowNum(p_rowNum)
        , m_colNum(p_colNum)
    {}
    
    Matrix(ContinuousMemory<ElementType, DeviceType> p_mem,
           size_t p_rowNum,
           size_t p_colNum)
        : m_mem(std::move(p_mem))
        , m_rowNum(p_rowNum)
        , m_colNum(p_colNum)
    {}

    bool operator== (const Matrix& val) const
    {
        return (m_mem == val.m_mem) &&
               (m_rowNum == val.m_rowNum) &&
               (m_colNum == val.m_colNum);
    }

    template <typename TOtherType>
    bool operator== (const TOtherType&) const
    {
        return false;
    }

    template <typename TData>
    bool operator!= (const TData& val) const
    {
        return !(operator==(val));
    }

    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }

    bool AvailableForWrite() const { return m_mem.UseCount() == 1; }

    void SetValue(size_t p_rowId, size_t p_colId, ElementType val)
    {
        assert(AvailableForWrite());
        assert((p_rowId < m_rowNum) && (p_colId < m_colNum));
        (m_mem.RawMemory())[p_rowId * m_colNum + p_colId] = val;
    }

    const auto operator () (size_t p_rowId, size_t p_colId) const
    {
        assert((p_rowId < m_rowNum) && (p_colId < m_colNum));
        return (m_mem.RawMemory())[p_rowId * m_colNum + p_colId];
    }

    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }
    
private:
    ContinuousMemory<ElementType, DeviceType> m_mem;
    size_t m_rowNum;
    size_t m_colNum;
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
