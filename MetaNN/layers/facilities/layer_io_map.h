#pragma once
#include <MetaNN/facilities/cont_metafuns/helpers.h>

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
}