#pragma once

/*
#include <MetaNN/data/dynamic.h>
#include <MetaNN/model/grad_col/grad_collector.h>
#include <stack>
#include <stdexcept>
*/
#include <type_traits>
#include <MetaNN/facilities/var_type_dict.h>

namespace MetaNN::LayerTraits
{
/*
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
*/
namespace NSLayerIOMapTrasfer
{
template <typename TVarTypeDict, typename... TKVs>
struct CreateVarTypeDict_
{
    using type = TVarTypeDict;
};

template <template<typename...> class TVarTypeDictCont, typename... TProcessed, typename TCur, typename... TKVs>
struct CreateVarTypeDict_<TVarTypeDictCont<TProcessed...>, TCur, TKVs...>
{
    using type = typename CreateVarTypeDict_<TVarTypeDictCont<TProcessed...,
                                                              typename TCur::KeyType>,
                                             TKVs...>::type;
};

template <typename TVarTypeDict>
auto FillVarTypeDict(TVarTypeDict&& curDict)
{
    return std::forward<TVarTypeDict>(curDict);
}

template <typename TVarTypeDict, typename TCur, typename... TKVs>
auto FillVarTypeDict(TVarTypeDict&& curDict)
{
    auto newDict = std::forward<TVarTypeDict>(curDict).template Set<typename TCur::KeyType>(std::declval<typename TCur::ValueType>());
    return FillVarTypeDict<decltype(newDict), TKVs...>(std::move(newDict));
}

template <typename TVarTypeDict, typename SeqCont>
struct VarTypeDict2IOMap_;

template <typename TVarTypeDict, int... IDs>
struct VarTypeDict2IOMap_<TVarTypeDict, ContMetaFun::Helper::IndexSequence<IDs...>>
{
    using type = LayerIOMap<LayerKV<typename TVarTypeDict::template KeyType<IDs>,
                                    typename TVarTypeDict::template ValueType<IDs>
                                   >...>;
};
}

template <typename TLayer, typename TLayerIOMap>
struct LayerIOMapTrasfer_;

template <typename TLayer, typename... TKVs>
struct LayerIOMapTrasfer_<TLayer, LayerIOMap<TKVs...>>
{
    using TVarTypeDictOri = NSLayerIOMapTrasfer::CreateVarTypeDict_<VarTypeDict<>, TKVs...>;
    using RVarTypeDictOri = typename TVarTypeDictOri::type;
    
    using RVarTypeDictCre = decltype(RVarTypeDictOri::Create());
    using TVarTypeDictFill = decltype(NSLayerIOMapTrasfer::FillVarTypeDict<RVarTypeDictCre, TKVs...>(RVarTypeDictOri::Create()));
    
    using TForwardRes = decltype(std::declval<TLayer>().FeedForward(std::declval<TVarTypeDictFill>()));
    
    using IndexSeq = ContMetaFun::Helper::MakeIndexSequence<TForwardRes::Length>;
    
    using type = typename NSLayerIOMapTrasfer::VarTypeDict2IOMap_<TForwardRes, IndexSeq>::type;
};

template <typename TLayer, typename TLayerIOMap>
using LayerIOMapTrasfer = typename LayerIOMapTrasfer_<TLayer, TLayerIOMap>::type;

template <bool IsAimDynamic, typename T>
auto DynamicTransWithFlag(T&& val)
{
	if constexpr (IsAimDynamic)
	{
		return MakeDynamic(std::forward<T>(val));
	}
	else
	{
		return std::forward<T>(val);
	}
}
}