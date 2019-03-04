#pragma once

#include <random>

namespace MetaNN
{
namespace NSInitializer
{
template <typename TElem, typename TDist, typename TEngine>
void FillWithDist(Matrix<TElem, DeviceTags::CPU>& data, TDist& dist, TEngine& engine)
{
    if (!data.AvailableForWrite())
    {
        throw std::runtime_error("Matrix is sharing weight, cannot fill-in.");
    }
    
    auto mem = LowerAccess(data);
    const size_t count = data.Shape().Count();
    auto r = mem.MutableRawMemory();
    
    for (size_t i = 0; i < count; ++i)
    {
        r[i] = (TElem)(dist(engine));
    }
}
}
}