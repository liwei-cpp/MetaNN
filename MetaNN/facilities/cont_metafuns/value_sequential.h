#pragma once
#include <MetaNN/facilities/cont_metafuns/helpers.h>

namespace MetaNN::ValueSequential
{
    template <typename TValueSeq, auto val>
    struct Contains_;
    
    template <template<auto...> class TValueCont, auto val, auto... vals>
    struct Contains_<TValueCont<vals...>, val>
    {
        constexpr static bool value = ((vals == val) || ...);
    };
    
    template <typename TValueSeq, auto val>
    constexpr static bool Contains = Contains_<TValueSeq, val>::value;

// Order===================================================================================
    namespace NSOrder
    {
        template <typename TIndexCont, typename TTypeCont>
        struct impl;

        template <template <auto...> typename TTypeCont, auto...TTypes, int...index>
        struct impl<Helper::IndexSequence<index...>, TTypeCont<TTypes...>>
            : Helper::KVBinder<Helper::Int_<TTypes>, Helper::Int_<index>> ...
        {
            using Helper::KVBinder<Helper::Int_<TTypes>, Helper::Int_<index>>::apply ...;
        };
    }

    template <typename TCon, auto TReq>
    struct Order_;

    template <template <auto...> typename TCon, auto... TParams, auto TReq>
    struct Order_<TCon<TParams...>, TReq>
    {
        using IndexSeq = Helper::MakeIndexSequence<sizeof...(TParams)>;
        using LookUpTable = NSOrder::impl<IndexSeq, TCon<TParams...>>;
        using ReqType = Helper::Int_<TReq>;
        using AimType = decltype(LookUpTable::apply((ReqType*)nullptr));
        constexpr static int value = AimType::value;
    };

    template <typename TCon, auto TReq>
    constexpr static int Order = Order_<TCon, TReq>::value;
//=========================================================================================
}