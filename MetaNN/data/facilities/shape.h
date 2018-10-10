#pragma once
#include <MetaNN/data/facilities/tags.h>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace MetaNN
{
template <typename T>
class Shape;

template <>
class Shape<CategoryTags::Scalar>
{
public:
    using ShapeCategory = CategoryTags::Scalar;
    
    constexpr size_t Count() const noexcept
    {
        return 1;
    }
    
    template <typename TOtherShape>
    bool operator== (const TOtherShape& val) const
    {
        static_assert(std::is_same_v<ShapeCategory, typename TOtherShape::ShapeCategory>);
        return Compare(val);
    }
    
protected:
    template <typename TOtherShape>
    bool Compare(const TOtherShape& val) const
    {
        return Count() == val.Count();
    }
};

template <>
class Shape<CategoryTags::Matrix>
{
public:
    using ShapeCategory = CategoryTags::Matrix;
    
public:
    Shape(size_t p_rowNum, size_t p_colNum)
        : m_rowNum(p_rowNum)
        , m_colNum(p_colNum)
    {}
    
public:
    size_t RowNum() const noexcept { return m_rowNum; }
    size_t ColNum() const noexcept { return m_colNum; }
    
    size_t Count() const noexcept
    {
        return m_rowNum * m_colNum;
    }
    
    template <typename TOtherShape>
    bool operator== (const TOtherShape& val) const
    {
        static_assert(std::is_same_v<ShapeCategory, typename TOtherShape::ShapeCategory>);
        return Compare(val);
    }
    
protected:
    template <typename TOtherShape>
    bool Compare(const TOtherShape& val) const
    {
        return (RowNum() == val.RowNum()) && (ColNum() == val.ColNum());
    }
    
private:
    size_t m_rowNum;
    size_t m_colNum;
};

template <>
class Shape<CategoryTags::ThreeDArray> : public Shape<CategoryTags::Matrix>
{
public:
    using ShapeCategory = CategoryTags::ThreeDArray;
    
public:
    Shape(size_t p_pageNum, size_t p_rowNum, size_t p_colNum)
        : Shape<CategoryTags::Matrix>(p_rowNum, p_colNum)
        , m_pageNum(p_pageNum)
    {}
    
public:
    size_t PageNum() const noexcept { return m_pageNum; }
    
    size_t Count() const noexcept
    {
        return m_pageNum * Shape<CategoryTags::Matrix>::Count();
    }
    
    template <typename TOtherShape>
    bool operator== (const TOtherShape& val) const
    {
        static_assert(std::is_same_v<ShapeCategory, typename TOtherShape::ShapeCategory>);
        return Compare(val);
    }
    
protected:
    template <typename TOtherShape>
    bool Compare(const TOtherShape& val) const
    {
        return (PageNum() == val.PageNum()) &&
               Shape<CategoryTags::Matrix>::Compare(val);
    }
    
private:
    size_t m_pageNum;
};

template <typename TSubCate>
class Shape<CategoryTags::Batch<TSubCate>> : public Shape<TSubCate>
{
public:
    using ShapeCategory = CategoryTags::Batch<TSubCate>;
    
public:
    template <typename...TParams>
    Shape(size_t p_batchNum, TParams&&... params)
        : Shape<TSubCate>(std::forward<TParams>(params)...)
        , m_batchNum(p_batchNum)
    {}
    
    Shape() = delete;
    
public:
    size_t BatchNum() const noexcept { return m_batchNum; }
    
    size_t Count() const noexcept
    {
        return BatchNum() * Shape<TSubCate>::Count();
    }
    
    template <typename TOtherShape>
    bool operator== (const TOtherShape& val) const
    {
        static_assert(std::is_same_v<ShapeCategory, typename TOtherShape::ShapeCategory>);
        return (BatchNum() == val.BatchNum()) && Compare(val);
    }
    
private:
    size_t m_batchNum;
};

template <typename TSubCate>
class Shape<CategoryTags::Sequence<TSubCate>> : public Shape<TSubCate>
{
public:
    using ShapeCategory = CategoryTags::Sequence<TSubCate>;
    
public:
    template <typename...TParams>
    Shape(size_t p_seqLen, TParams&&... params)
        : Shape<TSubCate>(std::forward<TParams>(params)...)
        , m_seqLen(p_seqLen)
    {}
    
    Shape() = delete;
    
public:
    size_t Length() const noexcept { return m_seqLen; }
    
    size_t Count() const noexcept
    {
        return Length() * Shape<TSubCate>::Count();
    }
    
    template <typename TOtherShape>
    bool operator== (const TOtherShape& val) const
    {
        static_assert(std::is_same_v<ShapeCategory, typename TOtherShape::ShapeCategory>);
        return (Length() == val.Length()) && Compare(val);
    }
    
private:
    size_t m_seqLen;
};

template <typename TSubCate>
class Shape<CategoryTags::BatchSequence<TSubCate>> : public Shape<TSubCate>
{
public:
    using ShapeCategory = CategoryTags::BatchSequence<TSubCate>;
    
public:
    template <typename TI, typename...TParams>
    Shape(TI b, TI e, TParams&&... params)
        : Shape<TSubCate>(std::forward<TParams>(params)...)
        , m_seqLenCont(b, e)
    {}
    
    Shape() = delete;
    
public:
    const auto& SeqLenContainer() const noexcept
    {
        return m_seqLenCont;
    }
    
    size_t Count() const noexcept
    {
        return std::accumulate(m_seqLenCont.begin(), m_seqLenCont.end(), 0) *
               Shape<TSubCate>::Count();
    }
    
    template <typename TOtherShape>
    bool operator== (const TOtherShape& val) const
    {
        static_assert(std::is_same_v<ShapeCategory, typename TOtherShape::ShapeCategory>);
        
        const auto& compCont = val.SeqLenContainer();
        if (m_seqLenCont.size() != compCont.size())
            return false;
        if (!std::equal(m_seqLenCont.begin(), m_seqLenCont.end(), compCont.begin()))
            return false;
        return Compare(val);
    }
    
private:
    std::vector<size_t> m_seqLenCont;
};
}