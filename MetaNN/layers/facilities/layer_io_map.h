#pragma once
#include <MetaNN/facilities/cont_metafuns/helpers.h>
#include <MetaNN/facilities/var_type_dict.h>

namespace MetaNN
{
    template <typename TKey, typename TValue>
    struct LayerKV : ContMetaFun::Helper::KVBinder<TKey, RemConstRef<TValue>>
    { };
    
    template <typename... TLayerKVs>
    struct LayerIOMap : TLayerKVs...
    {
        using TLayerKVs::apply...;
        static NullParameter apply(...);
        
        template <typename TKey>
        struct Find_
        {
            using type = decltype(LayerIOMap::apply((TKey*)nullptr));
        };
    
        template <typename TKey>
        using Find = typename Find_<TKey>::type;
    };
    
    template <typename TGradMap, typename... TKeys>
    struct FillGradMap_
    {
        using type = TGradMap;
    };
    
    template <typename ... TKeys>
    struct FillGradMap_<LayerIOMap<>, TKeys...>
    {
        using type = LayerIOMap<LayerKV<TKeys, NullParameter>...>;
    };
    
    template <typename TGradMap, typename... TKeys>
    using FillGradMap = typename FillGradMap_<TGradMap, TKeys...>::type;
    
    template <typename TMap>
    struct LayerContainer_;
    
    template <typename... TLayerKVs>
    struct LayerContainer_<LayerIOMap<TLayerKVs...>>
    {
        using type = VarTypeDict<typename TLayerKVs::KeyType ...>;
    };
    
    template <typename TMap>
    using LayerContainer = typename TMap::type;
    
    template <typename TLayer>
    auto LayerInputCont()
    {
        using TMap = typename TLayer::InputMap;
        using TCont = typename LayerContainer_<TMap>::type;
        return TCont::Create();
    }
    
    template <typename TLayer>
    auto LayerOutputCont()
    {
        using TMap = typename TLayer::GradMap;
        using TCont = typename LayerContainer_<TMap>::type;
        return TCont::Create();
    }
}