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
private:
    template <typename THead, typename...TRemain>
    bool SameDim(const THead&, const TRemain&...)
    {
        return true;
    }

    template <typename THead, typename TCur, typename...TRemain>
    bool SameDim(const THead& head, const TCur& cur, const TRemain&...rem)
    {
        const bool tmp = (head.RowNum() == cur.RowNum()) &&
                         (head.ColNum() == cur.ColNum());
        return tmp && SameDim(cur, rem...);
    }

public:
    template <typename THead, typename...TRemain>
    OperOrganizer(const THead& head, const TRemain&... rem)
        : m_rowNum(head.RowNum())
        , m_colNum(head.ColNum())
    {
        assert(SameDim(head, rem...));
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
private:
    template <typename THead, typename...TRemain>
    bool SameDim(const THead&, const TRemain&...)
    {
        return true;
    }

    template <typename THead, typename TCur, typename...TRemain>
    bool SameDim(const THead& head, const TCur& cur, const TRemain&...rem)
    {
        const bool tmp = (head.RowNum() == cur.RowNum()) &&
                         (head.ColNum() == cur.ColNum()) &&
                         (head.BatchNum() == cur.BatchNum());
        return tmp && SameDim(cur, rem...);
    }

public:
    template <typename THead, typename...TRemain>
    OperOrganizer(const THead& head, const TRemain&... rem)
        : m_rowNum(head.RowNum())
        , m_colNum(head.ColNum())
        , m_batchNum(head.BatchNum())
    {
        assert(SameDim(head, rem...));
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
private:
    template <typename THead, typename...TRemain>
    bool SameDim(const THead&, const TRemain&...)
    {
        return true;
    }

    template <typename THead, typename TCur, typename...TRemain>
    bool SameDim(const THead& head, const TCur& cur, const TRemain&...rem)
    {
        const bool tmp = (head.BatchNum() == cur.BatchNum());
        return tmp && SameDim(cur, rem...);
    }

public:
    template <typename THead, typename...TRemain>
    OperOrganizer(const THead& head, const TRemain&... rem)
        : m_batchNum(head.BatchNum())
    {
        assert(SameDim(head, rem...));
    }

    size_t BatchNum() const { return m_batchNum; }

private:
    size_t m_batchNum;
};
}