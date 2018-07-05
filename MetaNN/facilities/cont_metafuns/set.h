#pragma once
#include <MetaNN/facilities/cont_metafuns/helpers.h>
#include <MetaNN/facilities/cont_metafuns/sequential.h>

namespace MetaNN::ContMetaFun::Set
{
// HasKey =================================================================================
    template <typename TCon, typename TKey>
    struct HasKey_;
    
    template <template <typename...> typename TCon, typename TKey, typename... TValues>
    struct HasKey_<TCon<TValues...>, TKey>
    {
        constexpr static bool value = (std::is_same_v<TValues, TKey> || ...);
    };
    
    template <typename TCon, typename TKey>
    constexpr bool HasKey = HasKey_<TCon, TKey>::value;
//=========================================================================================

// Insert =================================================================================
    template <typename TCon, typename TKey, bool bMute = false>
    struct Insert_ : Sequential::PushBack_<TCon, TKey>
    {
        static_assert(!HasKey<TCon, TKey>);
    };
    
    template <typename TCon, typename TKey>
    struct Insert_<TCon, TKey, true>
    {
        using type = typename std::conditional_t<HasKey<TCon, TKey>,
                                                 Identity_<TCon>,
                                                 Sequential::PushBack_<TCon, TKey>>::type;
    };
    
    template <typename TCon, typename TKey, bool bMute = false>
    using Insert = typename Insert_<TCon, TKey, bMute>::type;
//=========================================================================================

// Erase ==================================================================================
namespace NSErase
{
    template <typename TCon, typename TKey, typename... TItems>
    struct Helper_
    {
        using type = TCon;
    };
    
    template <template <typename...> typename TCon, typename... TParams, typename TKey, typename TCur, typename... TItems>
    struct Helper_<TCon<TParams...>, TKey, TCur, TItems...>
    {
        using type = typename Helper_<TCon<TParams..., TCur>, TKey, TItems...>::type;
    };
    
    template <template <typename...> typename TCon, typename... TParams, typename TKey, typename... TItems>
    struct Helper_<TCon<TParams...>, TKey, TKey, TItems...>
    {
        using type = TCon<TParams..., TItems...>;
    };
}
    
    template <typename TCon, typename TKey>
    struct Erase_;
    
    template <template <typename...> typename TCon, typename TKey, typename... TItems>
    struct Erase_<TCon<TItems...>, TKey> : NSErase::Helper_<TCon<>, TKey, TItems...>
    {};
    
    template <typename TCon, typename TKey>
    using Erase = typename Erase_<TCon, TKey>::type;
//=========================================================================================
}
