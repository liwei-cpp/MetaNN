#pragma once

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
    const size_t rowNum = data.RowNum();
    const size_t colNum = data.ColNum();
    const size_t tgtPackNum = mem.RowLen();
    auto r = mem.MutableRawMemory();
    
    for (size_t i = 0; i < rowNum; ++i)
    {
        for (size_t j = 0; j < colNum; ++j)
        {
            r[j] = (TElem)(dist(engine));
        }
        r += tgtPackNum;
    }
}
}
}