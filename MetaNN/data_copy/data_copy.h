#pragma once

#include <MetaNN/data/_.h>
#include <stdexcept>

namespace MetaNN
{
template <typename TElem>
void DataCopy(const Matrix<TElem, DeviceTags::CPU>& src,
              Matrix<TElem, DeviceTags::CPU>& dst)
{
    if (src.Shape() != dst.Shape())
    {
        throw std::runtime_error("Error in data-copy: Matrix dimension mismatch.");
    }
    
    const auto mem_src = LowerAccess(src);
    auto mem_dst = LowerAccess(dst);

    const TElem* r1 = mem_src.RawMemory();
    TElem* r = mem_dst.MutableRawMemory();
        
    memcpy(r, r1, sizeof(TElem) * src.Shape().Count());
}
}