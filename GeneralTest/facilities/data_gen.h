#pragma once
#include <MetaNN/meta_nn.h>

template <typename TElem>
inline auto GenMatrix(std::size_t r, std::size_t c, TElem start = 0, TElem scale = 1)
{
    using namespace MetaNN;
    Matrix<TElem, MetaNN::DeviceTags::CPU> res(r, c);
    for (std::size_t i = 0; i < r; ++i)
    {
        for (std::size_t j = 0; j < c; ++j)
        {
            res.SetValue(i, j, (TElem)(start * scale));
            start += 1.0f;
        }
    }
    return res;
}

template <typename TElem>
inline auto GenBatchMatrix(size_t r, size_t c, size_t d, float start = 0, float scale = 1)
{
    using namespace MetaNN;
    Batch<TElem, MetaNN::DeviceTags::CPU, CategoryTags::Matrix> res(d, r, c);
    for (size_t i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < c; ++j)
        {
            for (size_t k = 0; k < d; ++k)
            {
                res.SetValue(k, i, j, (TElem)(start * scale));
                start += 1.0f;
            }
        }
    }
    return res;
}

template <typename TElem>
inline auto GenThreeDArray(size_t p, size_t r, size_t c, float start = 0, float scale = 1)
{
    using namespace MetaNN;
    ThreeDArray<TElem, MetaNN::DeviceTags::CPU> res(p, r, c);
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

template <typename TElem>
inline auto GenSequenceThreeDArray(size_t len, size_t p, size_t r, size_t c,
                                   float start = 0, float scale = 1)
{
    using namespace MetaNN;
    Sequence<TElem, DeviceTags::CPU, CategoryTags::ThreeDArray> res(len, p, r, c);
    for (size_t l = 0; l < len; ++l)
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