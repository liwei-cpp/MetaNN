#pragma once

namespace MetaNN::Helper
{
// Int_ ===========================================================================
    template <auto N>
    struct Int_
    {
        constexpr static auto value = N;
    };
// ================================================================================
// MakeIndexSequence ==============================================================
    template <int... I> struct IndexSequence;

    namespace NSIndexSequence
    {
        template <typename L, typename R> struct concat;

        template <int... L, int... R>
        struct concat<IndexSequence<L...>, IndexSequence<R...>>
        {
            using type = IndexSequence<L..., (R + sizeof...(L))...>;
        };
    }

    template <int N>
    struct MakeIndexSequence_
    {
        using type = typename NSIndexSequence::concat <
            typename MakeIndexSequence_<N / 2>::type,
            typename MakeIndexSequence_<N - N / 2>::type
        >::type;
    };

    template <>
    struct MakeIndexSequence_<1>
    {
        using type = IndexSequence<0>;
    };

    template <>
    struct MakeIndexSequence_<0>
    {
        using type = IndexSequence<>;
    };
    
    template <int N>
    using MakeIndexSequence = typename MakeIndexSequence_<N>::type;
// =========================================================================================

// KVBinder ================================================================================
    template <typename TK, typename TV>
    struct KVBinder
    {
        using KeyType = TK;
        using ValueType = TV;
        static TV apply(TK*);
    };
// =========================================================================================

// When ===========================================================================
    template <bool b>
    struct When;
// =========================================================================================

// Value Sequence =================================================================
    template <typename... TValues>
    struct ValueSequence;
// =========================================================================================

// Pair ====================================================================================
    template <typename V1, typename V2>
    struct Pair
    {
        using FirstType = V1;
        using SecondType = V2;
    };
// =========================================================================================
}
