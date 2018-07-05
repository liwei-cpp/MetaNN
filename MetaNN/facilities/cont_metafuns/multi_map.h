#pragma once
#include <MetaNN/facilities/cont_metafuns/helpers.h>
#include <MetaNN/facilities/cont_metafuns/sequential.h>

namespace MetaNN::ContMetaFun::MultiMap
{
// Insert =================================================================================
    namespace NSInsert
    {
        template <typename TProcessed, typename TKey, typename TValue, typename... TRemain>
        struct imp_
        {
            using type = Sequential::PushBack<TProcessed,
                                              MetaNN::ContMetaFun::Helper::Pair<TKey, TValue>>;
        };
        
        template <typename TProcessed,
                  typename TCur, typename... TRemain,
                  typename TKey, typename TValue>
        struct imp_<TProcessed, TKey, TValue, TCur, TRemain...>
                : imp_<Sequential::PushBack<TProcessed, TCur>, TKey, TValue, TRemain...>
        { };
        
        template <typename TProcessed,
                  typename TKey, typename TValue,
                  typename... TV, typename... TRemain>
        struct imp_<TProcessed, TKey, TValue,
                    Helper::Pair<TKey, TV...>, TRemain...>
        {
            using type = Sequential::PushBack<TProcessed,
                                              Helper::Pair<TKey, TV..., TValue>,
                                              TRemain...>;
        };
    }
    
    template <typename TCon, typename TKey, typename TValue>
    struct Insert_;
    
    template <template<typename...> typename TCon, typename... TItems, typename TKey, typename TValue>
    struct Insert_<TCon<TItems...>, TKey, TValue>
            : NSInsert::imp_<TCon<>, TKey, TValue, TItems...>
    {};
    
    template <typename TCon, typename TKey, typename TValue>
    using Insert = typename Insert_<TCon, TKey, TValue>::type;
    
// Find ===================================================================================
    namespace NSFind
    {
        template <typename TCon>
        struct map_;
        
        template <template <typename... > typename TCon, typename...TItem>
        struct map_<TCon<TItem...>> : ContMetaFun::Helper::KVBinder<typename TItem::KeyType, TItem>...
        {
            using ContMetaFun::Helper::KVBinder<typename TItem::KeyType, TItem>::apply...;
            static void apply(...);
        };
    }
    
    template <typename TCon, typename TKey>
    struct Find_
    {
        using type = decltype(NSFind::map_<TCon>::apply((TKey*)nullptr));
    };
    
    template <typename TCon, typename TKey>
    using Find = typename Find_<TCon, TKey>::type;
}
