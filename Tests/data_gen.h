#pragma once
#include <MetaNN/meta_nn2.h>
#include <cassert>

template <typename TElem, typename...TShapeParams>
inline auto GenTensor(TElem start, TElem step, TShapeParams... shapeParams)
{
    using namespace MetaNN;
    constexpr size_t dim = sizeof...(shapeParams);
    Tensor<TElem, MetaNN::DeviceTags::CPU, dim> res(shapeParams...);
    auto lowLayer = LowerAccess(res);
    auto mem = lowLayer.MutableRawMemory();
    for (size_t i = 0; i < res.Shape().Count(); ++i)
    {
        mem[i] = (TElem)start;
        start += step;
    }
    return res;
}

template <typename TElem, typename TIt, typename...TShapeParams>
inline auto FillTensor(TIt b, TShapeParams... shapeParams)
{
    using namespace MetaNN;
    constexpr size_t dim = sizeof...(shapeParams);
    Tensor<TElem, MetaNN::DeviceTags::CPU, dim> res(shapeParams...);
    auto lowLayer = LowerAccess(res);
    auto mem = lowLayer.MutableRawMemory();
    for (size_t i = 0; i < res.Shape().Count(); ++i)
    {
        mem[i] = (TElem)(*b);
        ++b;
    }
    return res;
}

template <typename TElem, size_t uDim>
bool Compare(const MetaNN::Tensor<TElem, MetaNN::DeviceTags::CPU, uDim>& v1,
             const MetaNN::Tensor<TElem, MetaNN::DeviceTags::CPU, uDim>& v2,
             TElem availGap)
{
    assert(v1.Shape() == v2.Shape());

    auto lowLayer1 = LowerAccess(v1);
    auto mem1 = lowLayer1.RawMemory();

    auto lowLayer2 = LowerAccess(v2);
    auto mem2 = lowLayer1.RawMemory();
    
    float diff = 0;
    for (size_t i = 0; i < v1.Shape().Count(); ++i)
    {
        float val = fabs(mem1[i] - mem2[i]);
        if (val > diff)
        {
            diff = val;
        }
    }
    
    return diff <= availGap;
}
