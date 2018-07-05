#pragma once

#include <MetaNN/data/facilities/traits.h>

namespace MetaNN
{
template <typename TOpTag, typename TCate>
class OperOrganizer;

template <typename TOpTag>
class OperOrganizer<TOpTag, CategoryTags::Scalar>
{
public:
    template <typename THead, typename...TRemain>
    OperOrganizer(const THead&, const TRemain&...)
    {}
};

template <typename TOpTag>
class OperOrganizer<TOpTag, CategoryTags::Matrix>
{
public:
    template <typename THead, typename...TRemain>
    OperOrganizer(const THead& head, const TRemain&... rem)
        : m_rowNum(head.RowNum())
        , m_colNum(head.ColNum())
    {
        assert(((m_rowNum == rem.RowNum()) && ...));
        assert(((m_colNum == rem.ColNum()) && ...));
    }

    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }

private:
    size_t m_rowNum;
    size_t m_colNum;
};

template <typename TOpTag>
class OperOrganizer<TOpTag, CategoryTags::BatchMatrix>
{
public:
    template <typename THead, typename...TRemain>
    OperOrganizer(const THead& head, const TRemain&... rem)
        : m_rowNum(head.RowNum())
        , m_colNum(head.ColNum())
        , m_batchNum(head.BatchNum())
    {
        assert(((m_rowNum == rem.RowNum()) && ...));
        assert(((m_colNum == rem.ColNum()) && ...));
        assert(((m_batchNum == rem.BatchNum()) && ...));
    }

    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }
    size_t BatchNum() const { return m_batchNum; }

private:
    size_t m_rowNum;
    size_t m_colNum;
    size_t m_batchNum;
};

template <typename TOpTag>
class OperOrganizer<TOpTag, CategoryTags::BatchScalar>
{
public:
    template <typename THead, typename...TRemain>
    OperOrganizer(const THead& head, const TRemain&... rem)
        : m_batchNum(head.BatchNum())
    {
        assert(((m_batchNum == rem.BatchNum()) && ...));
    }

    size_t BatchNum() const { return m_batchNum; }

private:
    size_t m_batchNum;
};
}