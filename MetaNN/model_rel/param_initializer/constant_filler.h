#pragma once

#include <random>
#include <stdexcept>
#include <type_traits>

namespace MetaNN
{
namespace NSConstantFiller
{
template <typename TElem>
void Fill(Matrix<TElem, DeviceTags::CPU>& mat, const double& val)
{
    if (!mat.AvailableForWrite())
    {
        throw std::runtime_error("Matrix is sharing weight, cannot fill-in.");
    }
    
    auto mem = LowerAccess(mat);
    const size_t rowNum = mat.RowNum();
    const size_t colNum = mat.ColNum();
    const size_t tgtPackNum = mem.RowLen();
    auto r = mem.MutableRawMemory();
    
    for (size_t i = 0; i < rowNum; ++i)
    {
        for (size_t j = 0; j < colNum; ++j)
        {
            r[j] = static_cast<TElem>(val);
        }
        r += tgtPackNum;
    }
}
}

class ConstantFiller
{
public:
    ConstantFiller(double val = 0)
        : m_value(val)
    {}
    
    template <typename TData>
    void Fill(TData& data, size_t /*fanin*/, size_t /*fanout*/)
    {
        NSConstantFiller::Fill(data, m_value);
    }
    
private:
    double m_value;
};
}