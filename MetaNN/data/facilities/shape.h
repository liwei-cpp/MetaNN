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
class Shape_;

template <>
class Shape_<CategoryTags::Scalar>
{
public:
    using ShapeCategory = CategoryTags::Scalar;
    
    constexpr size_t Count() const noexcept
    {
        return 1;
    }
    
    template <typename TOtherShape>
    bool operator== (const TOtherShape& val) const noexcept
    {
        static_assert(std::is_same_v<ShapeCategory, typename TOtherShape::ShapeCategory>);
        return Compare(val);
    }
    
    const Shape_<CategoryTags::Scalar>& Shape() const noexcept
    {
        return *this;
    }
    
protected:
    template <typename TOtherShape>
    bool Compare(const TOtherShape& val) const noexcept
    {
        return true;
    }
};

template <>
class Shape_<CategoryTags::Matrix>
{
public:
    using ShapeCategory = CategoryTags::Matrix;
    
public:
    Shape_(size_t p_rowNum, size_t p_colNum)
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
    bool operator== (const TOtherShape& val) const noexcept
    {
        static_assert(std::is_same_v<ShapeCategory, typename TOtherShape::ShapeCategory>);
        return Compare(val);
    }
    
    const Shape_<CategoryTags::Matrix>& Shape() const noexcept
    {
        return *this;
    }
    
protected:
    template <typename TOtherShape>
    bool Compare(const TOtherShape& val) const noexcept
    {
        return (RowNum() == val.RowNum()) && (ColNum() == val.ColNum());
    }
    
private:
    size_t m_rowNum;
    size_t m_colNum;
};

template <>
class Shape_<CategoryTags::ThreeDArray> : public Shape_<CategoryTags::Matrix>
{
public:
    using ShapeCategory = CategoryTags::ThreeDArray;
    
public:
    Shape_(size_t p_pageNum, size_t p_rowNum, size_t p_colNum)
        : Shape_<CategoryTags::Matrix>(p_rowNum, p_colNum)
        , m_pageNum(p_pageNum)
    {}
    
public:
    size_t PageNum() const noexcept { return m_pageNum; }
    
    size_t Count() const noexcept
    {
        return m_pageNum * Shape_<CategoryTags::Matrix>::Count();
    }
    
    template <typename TOtherShape>
    bool operator== (const TOtherShape& val) const noexcept
    {
        static_assert(std::is_same_v<ShapeCategory, typename TOtherShape::ShapeCategory>);
        return Compare(val);
    }
    
    const Shape_<CategoryTags::ThreeDArray>& Shape() const noexcept
    {
        return *this;
    }
    
protected:
    template <typename TOtherShape>
    bool Compare(const TOtherShape& val) const
    {
        return (PageNum() == val.PageNum()) &&
               Shape_<CategoryTags::Matrix>::Compare(val);
    }
    
private:
    size_t m_pageNum;
};

template <typename TSubCate>
class Shape_<CategoryTags::Batch<TSubCate>> : public Shape_<TSubCate>
{
public:
    using ShapeCategory = CategoryTags::Batch<TSubCate>;
    
public:
    template <typename...TParams>
    Shape_(size_t p_batchNum, TParams&&... params)
        : Shape_<TSubCate>(std::forward<TParams>(params)...)
        , m_batchNum(p_batchNum)
    {}
    
    Shape_() = delete;
    
public:
    size_t BatchNum() const noexcept { return m_batchNum; }
    
    size_t Count() const noexcept
    {
        return BatchNum() * Shape_<TSubCate>::Count();
    }
    
    template <typename TOtherShape>
    bool operator== (const TOtherShape& val) const noexcept
    {
        static_assert(std::is_same_v<ShapeCategory, typename TOtherShape::ShapeCategory>);
        return (BatchNum() == val.BatchNum()) && (Shape_<TSubCate>::Compare(val));
    }
    
    const Shape_<CategoryTags::Batch<TSubCate>>& Shape() const noexcept
    {
        return *this;
    }

private:
    size_t m_batchNum;
};

template <typename TSubCate>
class Shape_<CategoryTags::Sequence<TSubCate>> : public Shape_<TSubCate>
{
public:
    using ShapeCategory = CategoryTags::Sequence<TSubCate>;
    
public:
    template <typename...TParams>
    Shape_(size_t p_seqLen, TParams&&... params)
        : Shape_<TSubCate>(std::forward<TParams>(params)...)
        , m_seqLen(p_seqLen)
    {}
    
    Shape_() = delete;
    
public:
    size_t Length() const noexcept { return m_seqLen; }
    
    size_t Count() const noexcept
    {
        return Length() * Shape_<TSubCate>::Count();
    }
    
    template <typename TOtherShape>
    bool operator== (const TOtherShape& val) const noexcept
    {
        static_assert(std::is_same_v<ShapeCategory, typename TOtherShape::ShapeCategory>);
        return (Length() == val.Length()) && (Shape_<TSubCate>::Compare(val));
    }
    
    const Shape_<CategoryTags::Sequence<TSubCate>>& Shape() const noexcept
    {
        return *this;
    }
    
private:
    size_t m_seqLen;
};

template <typename TSubCate>
class Shape_<CategoryTags::BatchSequence<TSubCate>> : public Shape_<TSubCate>
{
public:
    using ShapeCategory = CategoryTags::BatchSequence<TSubCate>;
    
public:
    template <typename TI, typename...TParams>
    Shape_(TI b, TI e, TParams&&... params)
        : Shape_<TSubCate>(std::forward<TParams>(params)...)
        , m_seqLenCont(b, e)
    {}
    
    Shape_() = delete;
    
public:
    const auto& SeqLenContainer() const noexcept
    {
        return m_seqLenCont;
    }
    
    size_t Count() const noexcept
    {
        return std::accumulate(m_seqLenCont.begin(), m_seqLenCont.end(), 0) *
               Shape_<TSubCate>::Count();
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
        return Shape_<TSubCate>::Compare(val);
    }
    
    const Shape_<CategoryTags::BatchSequence<TSubCate>>& Shape() const noexcept
    {
        return *this;
    }
    
private:
    std::vector<size_t> m_seqLenCont;
};
}