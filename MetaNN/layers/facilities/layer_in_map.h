#pragma once
#include <MetaNN/facilities/cont_metafuns/helpers.h>
#include <MetaNN/facilities/cont_metafuns/map.h>
#include <MetaNN/facilities/cont_metafuns/sequential.h>
#include <MetaNN/facilities/var_type_dict.h>

namespace MetaNN
{
    template <typename... TPorts> struct LayerPortSet;
    
    template <typename TKey, typename TValue>
    struct LayerKV : Helper::KVBinder<TKey, RemConstRef<TValue>>
    { };
    
    template <typename... TLayerKVs>
    struct LayerInMap
    {
        template <typename TKey>
        using Find = Map::Find<LayerInMap, TKey, NullParameter>;
    };
    
    template <typename TLayerPorts>
    struct EmptyLayerInMap_;
    
    template <typename... TKeys>
    struct EmptyLayerInMap_<LayerPortSet<TKeys...>>
    {
        using type = LayerInMap<LayerKV<TKeys, NullParameter>...>;
    };
    
    template <typename TInMap>
    struct IsEmptyLayerInMap_;
    
    template <typename... TKVs>
    struct IsEmptyLayerInMap_<LayerInMap<TKVs...>>
    {
        constexpr static bool value = (std::is_same_v<typename TKVs::ValueType, NullParameter> && ...);
    };
    
    template <typename TInMap>
    constexpr bool IsEmptyLayerInMap = IsEmptyLayerInMap_<TInMap>::value;
    
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
    
    template <typename TInputMap, typename TKeySet>
    struct CheckInputMapAvailable_;

    template <typename... TKVs, typename TKeySet>
    struct CheckInputMapAvailable_<LayerInMap<TKVs...>, TKeySet>
    {
        constexpr static bool value1 = (sizeof...(TKVs) == Sequential::Size<TKeySet>);
        constexpr static bool value2 = (Set::HasKey<TKeySet, typename TKVs::KeyType> && ...);
        constexpr static bool value = value1 && value2;
    };
}