#pragma once
#include <MetaNN/meta_nn2.h>

template <typename TElem>
inline auto GenMatrix(std::size_t r, std::size_t c, TElem start = 0, TElem scale = 1)
{
    using namespace MetaNN;
    auto res = Matrix<TElem, MetaNN::DeviceTags::CPU>::CreateWithShape(r, c);
    for (std::size_t i = 0; i < r; ++i)
    {
        for (std::size_t j = 0; j < c; ++j)
        {
            res.SetValue((TElem)(start * scale), i, j);
            start += 1.0f;
        }
    }
    return res;
}

template <typename TElem>
inline auto GenThreeDArray(size_t p, size_t r, size_t c, float start = 0, float scale = 1)
{
    using namespace MetaNN;
    auto res = ThreeDArray<TElem, MetaNN::DeviceTags::CPU>::CreateWithShape(p, r, c);
    for (size_t k = 0; k < p; ++k)
    {
        for (size_t i = 0; i < r; ++i)
        {
            for (size_t j = 0; j < c; ++j)
            {
                res.SetValue((TElem)(start * scale), k, i, j);
                start += 1.0f;
            }
        }
    }
    return res;
}