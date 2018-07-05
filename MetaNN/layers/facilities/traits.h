#pragma once

#include <MetaNN/data/dynamic.h>
#include <MetaNN/model_rel/grad_col/grad_collector.h>
#include <stack>
#include <stdexcept>

namespace MetaNN
{
namespace LayerTraits
{
template <typename ElementType, typename DeviceType, typename CateType>
struct LayerInternalBufType_
{
    using tmp2 = DynamicData<ElementType, DeviceType, CateType>;
    using type = std::stack<tmp2, std::list<tmp2>>;
};

template <bool triger, bool batchMode, 
          typename ElementType, typename DeviceType,
          typename CateTypeSingle, typename CateTypeBatch>
struct LayerInternalBuf_
{
    using type = typename std::conditional_t<batchMode,
                                        LayerTraits::LayerInternalBufType_<ElementType, DeviceType,
                                                                           CateTypeBatch>,
                                        LayerTraits::LayerInternalBufType_<ElementType, DeviceType,
                                                                           CateTypeSingle>>::type;
};

template <bool batchMode, 
          typename ElementType, typename DeviceType,
          typename CateTypeSingle, typename CateTypeBatch>
struct LayerInternalBuf_<false, batchMode, ElementType, DeviceType, CateTypeSingle, CateTypeBatch>
{
    using type = NullParameter;
};

template <bool triger, bool batchMode, 
          typename ElementType, typename DeviceType,
          typename CateTypeSingle, typename CateTypeBatch>
using LayerInternalBuf = typename LayerInternalBuf_<triger, batchMode, ElementType, DeviceType, CateTypeSingle, CateTypeBatch>::type;

template <typename ElementType, typename DeviceType, typename CateType>
using LayerInternalBufType = typename LayerInternalBufType_<ElementType, DeviceType, CateType>::type;    

template <typename TWeight, typename TGrad, typename TGradCollector>
void MatrixGradCollect(const TWeight& weight,
                       TGrad& grad,
                       TGradCollector& col)
{
    while (!grad.empty())
    {
        auto g = grad.top();
        grad.pop();
        col.Collect(weight, g);
    }
}
}
}