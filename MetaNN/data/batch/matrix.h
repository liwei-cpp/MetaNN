#pragma once

#include <MetaNN/data/batch/batch.h>
#include <MetaNN/data/matrices/matrices.h>
#include <vector>

namespace MetaNN
{
template <typename TElem, typename TDevice>
struct LowerAccessImpl<Batch<TElem, TDevice, CategoryTags::Matrix>>;

template <typename TElement, typename TDevice>
class Batch<TElement, TDevice, CategoryTags::Matrix>
{
public:
    using ElementType = TElement;
    using DeviceType = TDevice;
    
    friend struct LowerAccessImpl<Batch<TElement, TDevice, CategoryTags::Matrix>>;
    
public:
    Batch(size_t p_batchNum = 0, size_t p_rowNum = 0, size_t p_colNum = 0)
        : m_mem(p_rowNum * p_colNum * p_batchNum)
        , m_rowNum(p_rowNum)
        , m_colNum(p_colNum)
        , m_batchNum(p_batchNum)
        , m_rowLen(p_colNum)
        , m_rawMatrixSize(p_rowNum * p_colNum)
    {}

    bool operator== (const Batch& val) const
    {
        return (m_mem == val.m_mem) &&
               (m_rowNum == val.m_rowNum) &&
               (m_colNum == val.m_colNum) &&
               (m_batchNum == val.m_batchNum) &&
               (m_rowLen == val.m_rowLen) &&
               (m_rawMatrixSize == val.m_rawMatrixSize);
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
    size_t BatchNum() const { return m_batchNum; }

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

    auto SubBatchMatrix(size_t p_rowB, size_t p_rowE, size_t p_colB, size_t p_colE) const
    {
        assert((p_rowB < m_rowNum) && (p_colB < m_colNum));
        assert((p_rowE <= m_rowNum) && (p_colE <= m_colNum));
        auto pos = m_mem.RawMemory() + p_rowB * m_rowLen + p_colB;
        return Batch(m_mem.SharedPtr(), pos,
                         p_rowE - p_rowB, p_colE - p_colB, m_batchNum,
                         m_rowLen, m_rawMatrixSize);
    }

    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }
    
private:
    Batch(std::shared_ptr<ElementType> p_mem,
              ElementType* p_memStart,
              size_t p_rowNum,
              size_t p_colNum,
              size_t p_batchNum,
              size_t p_rowLen,
              size_t p_matrixSize)
        : m_mem(p_mem, p_memStart)
        , m_rowNum(p_rowNum)
        , m_colNum(p_colNum)
        , m_batchNum(p_batchNum)
        , m_rowLen(p_rowLen)
        , m_rawMatrixSize(p_matrixSize)
    {}

private:
    ContinuousMemory<ElementType, DeviceType> m_mem;
    size_t m_rowNum;
    size_t m_colNum;
    size_t m_batchNum;
    size_t m_rowLen;
    size_t m_rawMatrixSize;
};

template <typename TElem, typename TDevice>
struct LowerAccessImpl<Batch<TElem, TDevice, CategoryTags::Matrix>>
{
    LowerAccessImpl(Batch<TElem, TDevice, CategoryTags::Matrix> p)
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
    Batch<TElem, TDevice, CategoryTags::Matrix> m_rawData;
};
}