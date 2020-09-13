#pragma once
#include <MetaNN/facilities/cont_metafuns/helpers.h>

namespace MetaNN::Sequential
{
// Create =================================================================================
namespace NSCreate
{
template <typename L, typename R> struct Concat;

template <template <typename...> typename TCont, typename... L, typename... R>
struct Concat<TCont<L...>, TCont<R...>>
{
    using type = TCont<L..., R...>;
};
}

template <template <typename...> typename TCont, typename TParam, int N>
struct Create_
{
    using type = typename NSCreate::Concat<
            typename Create_<TCont, TParam, N / 2>::type,
            typename Create_<TCont, TParam, N - N / 2>::type
        >::type;
};

template <template <typename...> typename TCont, typename TParam>
struct Create_<TCont, TParam, 1>
{
    using type = TCont<TParam>;
};

template <template <typename...> typename TCont, typename TParam>
struct Create_<TCont, TParam, 0>
{
    using type = TCont<>;
};

template <template <typename...> typename TCont, typename TParam, int N>
using Create = typename Create_<TCont, TParam, N>::type;
//=========================================================================================

// At =====================================================================================
namespace NSAt
{
template <typename ignore>
struct impl;

template <int... ignore>
struct impl<Helper::IndexSequence<ignore...>>
{
    template <typename nth>
    static nth apply(decltype(ignore, (void*)nullptr)..., nth*, ...);
};
}

template <typename TCon, int N>
struct At_;
    
template <template <typename...> typename TCon, typename... TParams, int N>
struct At_<TCon<TParams...>, N>
{
    using type = decltype(NSAt::impl<Helper::MakeIndexSequence<N>>::apply((TParams*)nullptr...));
};

template <typename TCon, int N>
using At = typename At_<TCon, N>::type;
//=========================================================================================

// Order===================================================================================
namespace NSOrder
{
template <typename TIndexCont, typename TTypeCont>
struct impl;

template <template <typename...> typename TTypeCont, typename...TTypes, int...index>
struct impl<Helper::IndexSequence<index...>, TTypeCont<TTypes...>>
    : Helper::KVBinder<TTypes, Helper::Int_<index>> ...
{
    using Helper::KVBinder<TTypes, Helper::Int_<index>>::apply ...;
};
}

template <typename TCon, typename TReq>
struct Order_;

template <template <typename...> typename TCon, typename... TParams, typename TReq>
struct Order_<TCon<TParams...>, TReq>
{
    using IndexSeq = Helper::MakeIndexSequence<sizeof...(TParams)>;
    using LookUpTable = NSOrder::impl<IndexSeq, TCon<TParams...>>;
    using AimType = decltype(LookUpTable::apply((TReq*)nullptr));
    constexpr static int value = AimType::value;
};

template <typename TCon, typename TReq>
constexpr static int Order = Order_<TCon, TReq>::value;
//=========================================================================================


// Set ====================================================================================
namespace NSSet
{
template <typename TCon, int N, typename TValue, typename TRemain, typename = Helper::When<true>>
struct imp_;

template <template <typename...> typename TCon, int N, typename TValue, typename... TProcessed, typename T0, typename... TRemain>
struct imp_<TCon<TProcessed...>, N, TValue, TCon<T0, TRemain...>, Helper::When<(N == 0)>>
{
    using type = TCon<TProcessed..., TValue, TRemain...>;
};

template <template <typename...> typename TCon, int N, typename TValue,
          typename... TProcessed, typename T0, typename T1, typename... TRemain>
struct imp_<TCon<TProcessed...>, N, TValue, TCon<T0, T1, TRemain...>, Helper::When<(N == 1)>>
{
    using type = TCon<TProcessed..., T0, TValue, TRemain...>;
};

template <template <typename...> typename TCon, int N, typename TValue,
          typename... TProcessed, typename T0, typename T1, typename T2, typename... TRemain>
struct imp_<TCon<TProcessed...>, N, TValue, TCon<T0, T1, T2, TRemain...>, Helper::When<(N == 2)>>
{
    using type = TCon<TProcessed..., T0, T1, TValue, TRemain...>;
};

template <template <typename...> typename TCon, int N, typename TValue,
          typename... TProcessed,
          typename T0, typename T1, typename T2, typename T3, typename... TRemain>
struct imp_<TCon<TProcessed...>, N, TValue, TCon<T0, T1, T2, T3, TRemain...>, Helper::When<(N == 3)>>
{
    using type = TCon<TProcessed..., T0, T1, T2, TValue, TRemain...>;
};

template <template <typename...> typename TCon, int N, typename TValue,
          typename... TProcessed,
          typename T0, typename T1, typename T2, typename T3, typename... TRemain>
struct imp_<TCon<TProcessed...>, N, TValue, TCon<T0, T1, T2, T3, TRemain...>, Helper::When<(N >= 4) && (N < 8)>>
{
    using type = typename imp_<TCon<TProcessed..., T0, T1, T2, T3>,
                               N - 4, TValue, TCon<TRemain...>>::type;
};

template <template <typename...> typename TCon, int N, typename TValue,
          typename... TProcessed,
          typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7,
          typename... TRemain>
struct imp_<TCon<TProcessed...>, N, TValue,
            TCon<T0, T1, T2, T3, T4, T5, T6, T7, TRemain...>, Helper::When<(N >= 8) && (N < 16)>>
{
    using type = typename imp_<TCon<TProcessed..., T0, T1, T2, T3, T4, T5, T6, T7>,
                               N - 8, TValue, TCon<TRemain...>>::type;
};

template <template <typename...> typename TCon, int N, typename TValue,
          typename... TProcessed,
          typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9, typename TA, typename TB,
          typename TC, typename TD, typename TE, typename TF,
          typename... TRemain>
struct imp_<TCon<TProcessed...>, N, TValue,
            TCon<T0, T1, T2, T3, T4, T5, T6, T7,
                 T8, T9, TA, TB, TC, TD, TE, TF,
                 TRemain...>, Helper::When<(N >= 16)>>
{
    using type = typename imp_<TCon<TProcessed...,
                                    T0, T1, T2, T3, T4, T5, T6, T7,
                                    T8, T9, TA, TB, TC, TD, TE, TF>,
                               N - 16, TValue, TCon<TRemain...>>::type;
};
}

template <typename TCont, int N, typename TValue>
struct Set_;

template <template <typename...> typename TCont, int N, typename TValue, typename... TParams>
struct Set_<TCont<TParams...>, N, TValue>
{
    using type = typename NSSet::imp_<TCont<>, N, TValue, TCont<TParams...>>::type;
};

template <typename TCont, int N, typename TValue>
using Set = typename Set_<TCont, N, TValue>::type;
//=========================================================================================

// Fold ===================================================================================
namespace NSFold
{
template <typename TState, template <typename, typename> typename F,
          typename... TRemain>
struct imp_
{
    using type = TState;
};

template <typename TState, template <typename, typename> typename F, typename T0>
struct imp_<TState, F, T0>
{
    using type = F<TState, T0>;
};

template <typename TState, template <typename, typename> typename F,
          typename T0, typename T1>
struct imp_<TState, F, T0, T1>
{
    using type = F<F<TState, T0>, T1>;
};

template <typename TState, template <typename, typename> typename F,
          typename T0, typename T1, typename T2>
struct imp_<TState, F, T0, T1, T2>
{
    using type = F<F<F<TState, T0>, T1>, T2>;
};

template <typename TState, template <typename, typename> typename F,
          typename T0, typename T1, typename T2, typename T3>
struct imp_<TState, F, T0, T1, T2, T3>
{
    using type = F<F<F<F<TState, T0>, T1>, T2>, T3>;
};

template <typename TState, template <typename, typename> typename F,
          typename T0, typename T1, typename T2, typename T3, typename T4>
struct imp_<TState, F, T0, T1, T2, T3, T4>
{
    using type = F<F<F<F<F<TState, T0>, T1>, T2>, T3>, T4>;
};

template <typename TState, template <typename, typename> typename F,
          typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
struct imp_<TState, F, T0, T1, T2, T3, T4, T5>
{
    using type = F<F<F<F<F<F<TState, T0>, T1>, T2>, T3>, T4>, T5>;
};

template <typename TState, template <typename, typename> typename F,
          typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename... TRemain>
struct imp_<TState, F, T0, T1, T2, T3, T4, T5, T6, TRemain...>
{
    using type = typename imp_<F<F<F<F<F<F<F<TState, T0>, T1>, T2>, T3>, T4>, T5>, T6>,
                               F, TRemain...>::type;
};
}

template <typename TInitState, typename TInputCont, template <typename, typename> typename F>
struct Fold_;

template <typename TInitState, template<typename...> typename TCont, typename... TParams,
          template <typename, typename> typename F>
struct Fold_<TInitState, TCont<TParams...>, F>
{
    template <typename S, typename I>
    using FF = typename F<S, I>::type;
    
    using type = typename NSFold::imp_<TInitState, FF, TParams...>::type;
};

template <typename TInitState, typename TInputCont, template <typename, typename> typename F>
using Fold = typename Fold_<TInitState, TInputCont, F>::type;
//=========================================================================================

// PushBack ===============================================================================
template <typename TCont, typename... TValue>
struct PushBack_;

template <template <typename...> typename TCont, typename... TParams, typename... TValue>
struct PushBack_<TCont<TParams...>, TValue...>
{
    using type = TCont<TParams..., TValue...>;
};

template <typename TCont, typename... TValue>
using PushBack = typename PushBack_<TCont, TValue...>::type;
//=========================================================================================

// Cascade ===============================================================================
template <typename TCont1, typename TCont2>
struct Cascade_;

template <template <typename...> typename TCont, typename... TParams1, typename... TParams2>
struct Cascade_<TCont<TParams1...>, TCont<TParams2...>>
{
    using type = TCont<TParams1..., TParams2...>;
};

template <typename TCont1, typename TCont2>
using Cascade = typename Cascade_<TCont1, TCont2>::type;
//=========================================================================================


// Transform ==============================================================================
template <typename TInCont, template <typename> typename F, template<typename...> typename TOutCont>
struct Transform_;

template <template <typename...> typename TInCont, typename... TInputs,
          template <typename> typename F, template<typename...> typename TOutCont>
struct Transform_<TInCont<TInputs...>, F, TOutCont>
{
    using type = TOutCont<typename F<TInputs>::type ...>;
};

template <typename TInCont, template <typename> typename F,
          template<typename...> typename TOutCont>
using Transform = typename Transform_<TInCont, F, TOutCont>::type;
//=========================================================================================
template <typename TArray>
struct Size_;

template <template <typename...> class Cont, typename...T>
struct Size_<Cont<T...>>
{
    constexpr static size_t value = sizeof...(T);
};

template <typename TArray>
constexpr static size_t Size = Size_<RemConstRef<TArray>>::value;

template <typename TSeqCont>
struct Head_;

template <template <typename...> class Container, typename TH, typename...TCases>
struct Head_<Container<TH, TCases...>>
{
    using type = TH;
};

template <typename TSeqCont>
using Head = typename Head_<TSeqCont>::type;

template <typename TSeqCont>
struct Tail_;

template <template <typename...> class Container, typename TH, typename...TCases>
struct Tail_<Container<TH, TCases...>>
{
    using type = Container<TCases...>;
};

template <typename TSeqCont>
using Tail = typename Tail_<TSeqCont>::type;
}