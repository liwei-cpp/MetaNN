#pragma once

#include <MetaNN/data/linear_table/linear_table.h>
#include <MetaNN/data/matrices/matrices.h>
#include <vector>

namespace MetaNN
{
template <typename TElement, typename TDevice>
class LinearTable<TElement, TDevice, CategoryTags::Matrix>
{
public:
    using ElementType = TElement;
    using DeviceType = TDevice;
    
    friend struct LowerAccessImpl<LinearTable<TElement, TDevice, CategoryTags::Matrix>>;
    
public:
    LinearTable(size_t p_batchNum = 0, size_t p_rowNum = 0, size_t p_colNum = 0)
        : m_mem(p_rowNum * p_colNum * p_batchNum)
        , m_rowNum(p_rowNum)
        , m_colNum(p_colNum)
        , m_batchNum(p_batchNum)
        , m_rowLen(p_colNum)
        , m_rawMatrixSize(p_rowNum * p_colNum)
    {}

    bool operator== (const LinearTable& val) const
    {
        return (m_mem == val.m_mem) &&
               (m_rowNum == val.m_rowNum) &&
               (m_colNum == val.m_colNum) &&
               (m_batchNum == val.m_batchNum) &&
               (m_rowLen == val.m_rowLen) &&
               (m_rawMatrixSize == val.m_rawMatrixSize);
    }

    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }

    bool AvailableForWrite() const { return m_mem.UseCount() == 1; }

    void SetValue(size_t p_batchId, size_t p_rowId, size_t p_colId, ElementType val)
    {
        assert(AvailableForWrite());
        assert((p_rowId < m_rowNum) &&
               (p_colId < m_colNum) &&
               (p_batchId < m_batchNum));
        
        size_t pos = p_batchId * m_rawMatrixSize + p_rowId * m_rowLen + p_colId;
        (m_mem.RawMemory())[pos] = val;
    }

    const auto operator [] (size_t p_batchId) const
    {
        assert(p_batchId < m_batchNum);
        
        auto pos = m_mem.RawMemory() + p_batchId * m_rawMatrixSize;
        return Matrix<TElement, TDevice>(m_mem.SharedPtr(), pos,
                                         m_rowNum, m_colNum, m_rowLen);
    }
    
protected:
    size_t Count() const { return m_batchNum; }
    
private:
    ContinuousMemory<ElementType, DeviceType> m_mem;
    size_t m_rowNum;
    size_t m_colNum;
    size_t m_batchNum;
    size_t m_rowLen;
    size_t m_rawMatrixSize;
};

template <typename TElem, typename TDevice>
struct LowerAccessImpl<LinearTable<TElem, TDevice, CategoryTags::Matrix>>
{
    LowerAccessImpl(LinearTable<TElem, TDevice, CategoryTags::Matrix> p)
        : m_rawData(std::move(p))
    {}

    auto MutableRawMemory()
    {
        return m_rawData.m_mem.RawMemory();
    }

    const auto RawMemory() const
    {
        return m_rawData.m_mem.RawMemory();
    }

    size_t RowLen() const
    {
        return m_rawData.m_rowLen;
    }
    
    size_t RawMatrixSize() const
    {
        return m_rawData.m_rawMatrixSize;
    }

private:
    LinearTable<TElem, TDevice, CategoryTags::Matrix> m_rawData;
};
}