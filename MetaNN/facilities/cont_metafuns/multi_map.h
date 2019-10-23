#pragma once
#include <MetaNN/facilities/cont_metafuns/helpers.h>
#include <MetaNN/facilities/cont_metafuns/sequential.h>

namespace MetaNN::MultiMap
{
// Insert =================================================================================
    namespace NSInsert
    {
        template <typename TProcessed, typename TKey, typename TValue, typename... TRemain>
        struct imp_
        {
            using Value = Helper::ValueSequence<TValue>;
            using type = Sequential::PushBack<TProcessed,
                                              Helper::KVBinder<TKey, Value>>;
        };
        
        template <typename TProcessed,
                  typename TCur, typename... TRemain,
                  typename TKey, typename TValue>
        struct imp_<TProcessed, TKey, TValue, TCur, TRemain...>
                : imp_<Sequential::PushBack<TProcessed, TCur>, TKey, TValue, TRemain...>
        { };
        
        template <typename TProcessed,
                  typename TKey, typename TValue,
                  typename TVs, typename... TRemain>
        struct imp_<TProcessed, TKey, TValue,
                    Helper::KVBinder<TKey, TVs>, TRemain...>
        {
            
            using type = Sequential::PushBack<TProcessed,
                                              Helper::KVBinder<TKey, Sequential::PushBack<TVs, TValue>>,
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
        struct map_<TCon<TItem...>> : TItem...
        {
            using TItem::apply...;
            static Helper::ValueSequence<> apply(...);
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
