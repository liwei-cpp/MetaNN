#pragma once
#include <MetaNN/facilities/cont_metafuns/helpers.h>
#include <MetaNN/facilities/var_type_dict.h>

namespace MetaNN
{
    template <typename... TPorts> struct LayerPortSet;
    
    template <typename TLayer>
    struct LayerInputPortSet_
    {
        using type = LayerPortSet<struct LayerInput>;
    };
    
    template <template <typename, typename> class TLayer>
    using LayerInputPortSet = typename LayerInputPortSet_<TLayer<void, void>>::type;
    
    template <typename TLayer>
    struct LayerOutputPortSet_
    {
        using type = LayerPortSet<struct LayerOutput>;
    };
    
    template <template <typename, typename> class TLayer>
    using LayerOutputPortSet = typename LayerOutputPortSet_<TLayer<void, void>>::type;
    
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
    
    template <typename TLayerPorts>
    struct EmptyLayerIOMap_;
    
    template <typename... TKeys>
    struct EmptyLayerIOMap_<LayerPortSet<TKeys...>>
    {
        using type = LayerIOMap<LayerKV<TKeys, NullParameter>...>;
    };
    
    template <typename TLayerPorts>
    using EmptyLayerIOMap = typename EmptyLayerIOMap_<TLayerPorts>::type;
    
    template <typename TIoMap>
    struct IsEmptyLayerIOMap_;
    
    template <typename... TKVs>
    struct IsEmptyLayerIOMap_<LayerIOMap<TKVs...>>
    {
        constexpr static bool value = (std::is_same_v<typename TKVs::ValueType, NullParameter> && ...);
    };
    
    template <typename TIoMap>
    constexpr bool IsEmptyLayerIOMap = IsEmptyLayerIOMap_<TIoMap>::value;
    
    template <typename TSet>
    struct LayerContainer_;
    
    template <typename... TPorts>
    struct LayerContainer_<LayerPortSet<TPorts...>>
    {
        using type = VarTypeDict<TPorts ...>;
    };
    
    template <typename TLayer>
    auto LayerInputCont()
    {
        using TMap = typename TLayer::InputPortSet;
        using TCont = typename LayerContainer_<TMap>::type;
        return TCont::Create();
    }
    
    template <typename TLayer>
    auto LayerOutputCont()
    {
        using TMap = typename TLayer::OutputPortSet;
        using TCont = typename LayerContainer_<TMap>::type;
        return TCont::Create();
    }
}