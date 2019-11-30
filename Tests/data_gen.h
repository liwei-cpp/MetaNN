#pragma once
#include <MetaNN/meta_nn.h>
#include <cassert>

template <typename TElem>
inline auto GenMatrix(std::size_t r, std::size_t c, TElem start = 0, TElem step = 1)
{
    using namespace MetaNN;
    Matrix<TElem, MetaNN::DeviceTags::CPU> res(r, c);
    
    TElem cur{};
    for (std::size_t i = 0; i < r; ++i)
    {
        for (std::size_t j = 0; j < c; ++j)
        {
            res.SetValue(i, j, (TElem)(start + cur));
            cur += step;
        }
    }
    return res;
}

template <typename TElem, typename TIt>
inline auto FillMatrix(std::size_t r, std::size_t c, TIt b)
{
    using namespace MetaNN;
    Matrix<TElem, MetaNN::DeviceTags::CPU> res(r, c);
    
    for (std::size_t i = 0; i < r; ++i)
    {
        for (std::size_t j = 0; j < c; ++j)
        {
            res.SetValue(i, j, (TElem)(*b));
            ++b;
        }
    }
    return res;
}

template <typename TElem>
inline auto GenThreeDArray(size_t p, size_t r, size_t c, TElem start = 0, TElem step = 1)
{
    using namespace MetaNN;
    ThreeDArray<TElem, MetaNN::DeviceTags::CPU> res(p, r, c);
    
    TElem cur{};
    for (size_t k = 0; k < p; ++k)
    {
        for (size_t i = 0; i < r; ++i)
        {
            for (size_t j = 0; j < c; ++j)
            {
                res.SetValue(k, i, j, (TElem)(start + cur));
                cur += step;
            }
        }
    }
    return res;
}

template <typename TElem>
inline auto GenBatchScalar(size_t len, TElem start = 0, TElem step = 1)
{
    using namespace MetaNN;
    BatchScalar<TElem, MetaNN::DeviceTags::CPU> res(len);
    TElem cur{};
    for (size_t k = 0; k < len; ++k)
    {
        res.SetValue(k, (TElem)(start + cur));
        cur += step;
    }
    return res;
}

template <typename TElem>
inline auto GenBatchMatrix(size_t p, size_t r, size_t c, TElem start = 0, TElem step = 1)
{
    using namespace MetaNN;
    BatchMatrix<TElem, MetaNN::DeviceTags::CPU> res(p, r, c);
    
    TElem cur{};
    for (size_t k = 0; k < p; ++k)
    {
        for (size_t i = 0; i < r; ++i)
        {
            for (size_t j = 0; j < c; ++j)
            {
                res.SetValue(k, i, j, (TElem)(start + cur));
                cur += step;
            }
        }
    }
    return res;
}

template <typename TElem>
inline auto GenBatchThreeDArray(size_t b, size_t p, size_t r, size_t c, TElem start = 0, TElem step = 1)
{
    using namespace MetaNN;
    BatchThreeDArray<TElem, MetaNN::DeviceTags::CPU> res(b, p, r, c);
    TElem cur{};
    for (size_t l = 0; l < b; ++l)
    {
        for (size_t k = 0; k < p; ++k)
        {
            for (size_t i = 0; i < r; ++i)
            {
                for (size_t j = 0; j < c; ++j)
                {
                    res.SetValue(l, k, i, j, (TElem)(start + cur));
                    cur += step;
                }
            }
        }
    }
    return res;
}

template <typename TElem>
inline auto GenScalarSequence(size_t len, TElem start = 0, TElem scale = 1)
{
    using namespace MetaNN;
    ScalarSequence<TElem, MetaNN::DeviceTags::CPU> res(len);
    for (size_t k = 0; k < len; ++k)
    {
        res.SetValue(k, (TElem)(start * scale));
        start += 1.0f;
    }
    return res;
}

template <typename TElem>
inline auto GenMatrixSequence(size_t p, size_t r, size_t c, TElem start = 0, TElem scale = 1)
{
    using namespace MetaNN;
    MatrixSequence<TElem, MetaNN::DeviceTags::CPU> res(p, r, c);
    for (size_t k = 0; k < p; ++k)
    {
        for (size_t i = 0; i < r; ++i)
        {
            for (size_t j = 0; j < c; ++j)
            {
                res.SetValue(k, i, j, (TElem)(start * scale));
                start += 1.0f;
            }
        }
    }
    return res;
}

template <typename TElem, typename TIt>
inline auto FillMatrixSequence(size_t p, size_t r, size_t c, TIt b)
{
    using namespace MetaNN;
    MatrixSequence<TElem, DeviceTags::CPU> res(p, r, c);
    for (size_t k = 0; k < p; ++k)
    {
        for (size_t i = 0; i < r; ++i)
        {
            for (size_t j = 0; j < c; ++j)
            {
                res.SetValue(k, i, j, (TElem)(*b));
                ++b;
            }
        }
    }
    return res;
}

template <typename TElem>
inline auto GenThreeDArraySequence(size_t b, size_t p, size_t r, size_t c, TElem start = 0, TElem scale = 1)
{
    using namespace MetaNN;
    ThreeDArraySequence<TElem, DeviceTags::CPU> res(b, p, r, c);
    for (size_t l = 0; l < b; ++l)
    {
        for (size_t k = 0; k < p; ++k)
        {
            for (size_t i = 0; i < r; ++i)
            {
                for (size_t j = 0; j < c; ++j)
                {
                    res.SetValue(l, k, i, j, (TElem)(start * scale));
                    start += 1.0f;
                }
            }
        }
    }
    return res;
}

template <typename TElem, typename TLen>
inline auto GenBatchScalarSequence(const std::vector<TLen>& seqs, TElem start = 0, TElem scale = 1)
{
    using namespace MetaNN;
    BatchScalarSequence<TElem, MetaNN::DeviceTags::CPU> res(seqs);
    for (size_t i = 0; i < seqs.size(); ++i)
    {
        for (size_t k = 0; k < (size_t)seqs[i]; ++k)
        {
            res.SetValue(i, k, (TElem)(start * scale));
            start += 1.0f;
        }
    }
    return res;
}

template <typename TElem, typename TLen>
inline auto GenBatchMatrixSequence(const std::vector<TLen>& seqs, size_t rowNum, size_t colNum, TElem start = 0, TElem scale = 1)
{
    using namespace MetaNN;
    BatchMatrixSequence<TElem, MetaNN::DeviceTags::CPU> res(seqs, rowNum, colNum);
    for (size_t i = 0; i < seqs.size(); ++i)
    {
        for (size_t k = 0; k < (size_t)seqs[i]; ++k)
        {
            for (size_t r = 0; r < rowNum; ++r)
            {
                for (size_t c = 0; c < colNum; ++c)
                {
                    res.SetValue(i, k, r, c, (TElem)(start * scale));
                    start += 1.0f;                
                }
            }
        }
    }
    return res;
}

template <typename TElem, typename TLen>
inline auto GenBatchThreeDArraySequence(const std::vector<TLen>& seqs, size_t pageNum, size_t rowNum, size_t colNum, TElem start = 0, TElem scale = 1)
{
    using namespace MetaNN;
    BatchThreeDArraySequence<TElem, MetaNN::DeviceTags::CPU> res(seqs, pageNum, rowNum, colNum);
    for (size_t i = 0; i < seqs.size(); ++i)
    {
        for (size_t k = 0; k < (size_t)seqs[i]; ++k)
        {
            for (size_t p = 0; p < pageNum; ++p)
            {
                for (size_t r = 0; r < rowNum; ++r)
                {
                    for (size_t c = 0; c < colNum; ++c)
                    {
                        res.SetValue(i, k, p, r, c, (TElem)(start * scale));
                        start += 1.0f;
                    }
                }
            }
        }
    }
    return res;
}

template <typename TElem>
bool Compare(const MetaNN::Matrix<TElem, MetaNN::DeviceTags::CPU>& v1,
             const MetaNN::Matrix<TElem, MetaNN::DeviceTags::CPU>& v2, TElem availGap)
{
    assert(v1.Shape() == v2.Shape());

    float diff = 0;
    for (size_t i = 0; i < v1.Shape().RowNum(); ++i)
    {
        for (size_t j = 0; j < v1.Shape().ColNum(); ++j)
        {
            float val = fabs(v1(i, j) - v2(i, j));
            if (val > diff)
            {
                diff = val;
            }
        }
    }
    
    return diff <= availGap;
}

template <typename TElem>
bool Compare(const MetaNN::MatrixSequence<TElem, MetaNN::DeviceTags::CPU>& v1,
             const MetaNN::MatrixSequence<TElem, MetaNN::DeviceTags::CPU>& v2, TElem availGap)
{
    assert(v1.Shape() == v2.Shape());

    float diff = 0;
    for (size_t k = 0; k < v1.Shape().Length(); ++k)
    {
        for (size_t i = 0; i < v1.Shape().RowNum(); ++i)
        {
            for (size_t j = 0; j < v1.Shape().ColNum(); ++j)
            {
                float val = fabs(v1[k](i, j) - v2[k](i, j));
                if (val > diff)
                {
                    diff = val;
                }
            }
        }
    }
    return diff <= availGap;
}