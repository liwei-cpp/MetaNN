#pragma once

#include <MetaNN/data/linear_table/linear_table.h>
#include <MetaNN/data/3d_array/3d_array.h>
#include <vector>

namespace MetaNN
{
template <typename TElement, typename TDevice>
class LinearTable<TElement, TDevice, CategoryTags::ThreeDArray>
{
public:
    using ElementType = TElement;
    using DeviceType = TDevice;
    
    friend struct LowerAccessImpl<LinearTable<TElement, TDevice, CategoryTags::ThreeDArray>>;
    
public:
    LinearTable()
        : LinearTable(0, 0, 0, 0)
    {}
    
    LinearTable(size_t p_batchNum, size_t p_pageNum, size_t p_rowNum, size_t p_colNum)
        : m_mem(p_pageNum * p_rowNum * p_colNum * p_batchNum)
        , m_pageNum(p_pageNum)
        , m_rowNum(p_rowNum)
        , m_colNum(p_colNum)
        , m_batchNum(p_batchNum)
    {}

    bool operator== (const LinearTable& val) const
    {
        return (m_mem == val.m_mem) &&
               (m_pageNum == val.m_pageNum) &&
               (m_rowNum == val.m_rowNum) &&
               (m_colNum == val.m_colNum) &&
               (m_batchNum == val.m_batchNum);
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

    size_t PageNum() const { return m_pageNum; }
    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }

    bool AvailableForWrite() const { return m_mem.UseCount() == 1; }

    void SetValue(size_t p_batchId, size_t p_pageId, size_t p_rowId, size_t p_colId, ElementType val)
    {
        assert(AvailableForWrite());
        assert((p_pageId < m_pageNum) &&
               (p_rowId < m_rowNum) &&
               (p_colId < m_colNum) &&
               (p_batchId < m_batchNum));
        
        size_t pos = ((p_batchId * m_pageNum + p_pageId) * m_rowNum + p_rowId) * m_colNum + p_colId;
        (m_mem.RawMemory())[pos] = val;
    }

    const auto operator [] (size_t p_batchId) const
    {
        assert(p_batchId < m_batchNum);
        
        auto pos = p_batchId * m_pageNum * m_rowNum * m_colNum;
        return ThreeDArray<TElement, TDevice>(m_mem.Shift(pos),
                                              m_pageNum, m_rowNum, m_colNum);
    }
    
protected:
    size_t Count() const { return m_batchNum; }
private:
    ContinuousMemory<ElementType, DeviceType> m_mem;
    size_t m_pageNum;
    size_t m_rowNum;
    size_t m_colNum;
    size_t m_batchNum;
};

template <typename TElem, typename TDevice>
struct LowerAccessImpl<LinearTable<TElem, TDevice, CategoryTags::ThreeDArray>>
{
    LowerAccessImpl(LinearTable<TElem, TDevice, CategoryTags::ThreeDArray> p)
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

private:
    LinearTable<TElem, TDevice, CategoryTags::ThreeDArray> m_rawData;
};
}