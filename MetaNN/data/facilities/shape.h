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
namespace NSShape
{
template <typename T>
constexpr bool IsCardinalTag = std::is_same_v<T, CategoryTags::Scalar> ||
                               std::is_same_v<T, CategoryTags::Matrix> ||
                               std::is_same_v<T, CategoryTags::ThreeDArray>;
}

template <typename T>
class Shape;

template <>
class Shape<CategoryTags::Scalar>
{
public:
    explicit Shape() = default;
    constexpr size_t Count() const noexcept
    {
        return 1;
    }
    
    bool operator== (const Shape&) const noexcept
    {
        return true;
    }    
};

template <>
class Shape<CategoryTags::Matrix>
{
public:
    explicit Shape(size_t p_rowNum = 0, size_t p_colNum = 0)
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
    
    bool operator== (const Shape& val) const noexcept
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
    explicit Shape(size_t p_pageNum = 0, size_t p_rowNum = 0, size_t p_colNum = 0)
        : Shape<CategoryTags::Matrix>(p_rowNum, p_colNum)
        , m_pageNum(p_pageNum)
    {}
    
public:
    size_t PageNum() const noexcept { return m_pageNum; }
    
    size_t Count() const noexcept
    {
        return m_pageNum * Shape<CategoryTags::Matrix>::Count();
    }
    
    bool operator== (const Shape& val) const noexcept
    {
        return (PageNum() == val.PageNum()) &&
               Shape<CategoryTags::Matrix>::operator==(val);
    }
    
private:
    size_t m_pageNum;
};

template <typename TSubCate>
class Shape<CategoryTags::Batch<TSubCate>> : public Shape<TSubCate>
{
    static_assert(NSShape::IsCardinalTag<TSubCate>);
    
public:
    template <typename...TParams>
    explicit Shape(size_t p_batchNum = 0, TParams&&... params)
        : Shape<TSubCate>(std::forward<TParams>(params)...)
        , m_batchNum(p_batchNum)
    {}
    
public:
    size_t BatchNum() const noexcept { return m_batchNum; }
    
    size_t Count() const noexcept
    {
        return BatchNum() * Shape<TSubCate>::Count();
    }
    
    bool operator== (const Shape& val) const noexcept
    {
        return (BatchNum() == val.BatchNum()) &&
               (Shape<TSubCate>::operator==(val));
    }

private:
    size_t m_batchNum;
};

template <typename TSubCate>
class Shape<CategoryTags::Sequence<TSubCate>> : public Shape<TSubCate>
{
    static_assert(NSShape::IsCardinalTag<TSubCate>);
    
public:
    template <typename...TParams>
    explicit Shape(size_t p_seqLen = 0, TParams&&... params)
        : Shape<TSubCate>(std::forward<TParams>(params)...)
        , m_seqLen(p_seqLen)
    {}
    
public:
    size_t Length() const noexcept { return m_seqLen; }
    
    size_t Count() const noexcept
    {
        return Length() * Shape<TSubCate>::Count();
    }
    
    bool operator== (const Shape& val) const noexcept
    {
        return (Length() == val.Length()) && 
               (Shape<TSubCate>::operator==(val));
    }
    
private:
    size_t m_seqLen;
};

template <typename TSubCate>
class Shape<CategoryTags::BatchSequence<TSubCate>> : public Shape<TSubCate>
{
    static_assert(NSShape::IsCardinalTag<TSubCate>);
    
public:
    template <typename TI, typename...TParams>
    explicit Shape(TI b, TI e, TParams&&... params)
        : Shape<TSubCate>(std::forward<TParams>(params)...)
        , m_seqLenCont(b, e)
    {}
    
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
    
    bool operator== (const Shape& val) const
    {
        return (m_seqLenCont == val.m_seqLenCont) &&
               (Shape<TSubCate>::operator==(val));
    }
    
private:
    std::vector<size_t> m_seqLenCont;
};
}