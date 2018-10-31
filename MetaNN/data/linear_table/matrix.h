#pragma once

#include <MetaNN/data/cardinal/matrix/matrix_base.h>
#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/data/facilities/continuous_memory.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/data/linear_table/linear_table_base.h>
#include <cassert>

namespace MetaNN
{
template <typename TElement, template <typename> class TCateWrapper>
class LinearTable<TElement, DeviceTags::CPU, TCateWrapper<CategoryTags::Matrix>>
    : public Shape_<TCateWrapper<CategoryTags::Matrix>>
{
public:
    using ElementType = TElement;
    using DeviceType = DeviceTags::CPU;
    
    friend struct LowerAccessImpl<LinearTable<TElement, DeviceTags::CPU, CategoryTags::Matrix>>;
    
public:
    LinearTable(size_t p_batchNum = 0, size_t p_rowNum = 0, size_t p_colNum = 0)
        : Shape_<TCateWrapper<CategoryTags::Matrix>>(p_batchNum, p_rowNum, p_colNum)
        , m_mem(p_rowNum * p_colNum * p_batchNum)
    {}

    bool operator== (const LinearTable& val) const
    {
        using TShape = Shape_<TCateWrapper<CategoryTags::Matrix>>;
        return (TShape::Shape() == val.Shape()) &&
               (m_mem == val.m_mem);
    }
    
    template <typename TOtherType,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<TOtherType>, LinearTable>>>
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

    void SetValue(size_t p_batchId, size_t p_rowId, size_t p_colId, ElementType val)
    {
        using TShape = Shape_<TCateWrapper<CategoryTags::Matrix>>;
        const size_t rowNum = TShape::RowNum();
        const size_t colNum = TShape::ColNum();
        const auto& wrapperDim = NSLinearTable::WrapperDim(TShape::Shape());
        
        assert(AvailableForWrite());
        assert((p_rowId < rowNum) &&
               (p_colId < colNum) &&
               (p_batchId < wrapperDim));
        
        size_t pos = (p_batchId * rowNum + p_rowId) * colNum + p_colId;
        (m_mem.RawMemory())[pos] = val;
    }

    const auto operator [] (size_t p_batchId) const
    {
        using TShape = Shape_<TCateWrapper<CategoryTags::Matrix>>;
        const size_t rowNum = TShape::RowNum();
        const size_t colNum = TShape::ColNum();
        const auto& wrapperDim = NSLinearTable::WrapperDim(TShape::Shape());
        
        assert(p_batchId < wrapperDim);
        
        auto pos = p_batchId * rowNum * colNum;
        return Matrix<ElementType, DeviceType>(m_mem.Shift(pos), rowNum, colNum);
    }
    
    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }
    
private:
    ContinuousMemory<ElementType, DeviceType> m_mem;
};

template <typename TElem, template <typename> class TCateWrapper>
struct LowerAccessImpl<LinearTable<TElem, DeviceTags::CPU, TCateWrapper<CategoryTags::Matrix>>>
{
    LowerAccessImpl(LinearTable<TElem, DeviceTags::CPU, TCateWrapper<CategoryTags::Matrix>> p)
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
    LinearTable<TElem, DeviceTags::CPU, TCateWrapper<CategoryTags::Matrix>> m_rawData;
};
}