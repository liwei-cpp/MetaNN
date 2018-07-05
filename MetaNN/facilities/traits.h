#pragma once

#include <cstddef>

namespace MetaNN
{
template <typename T>
struct Identity_
{
    using type = T;
};

// Or
template <bool cur, typename TNext>
constexpr static bool OrValue = true;

template <typename TNext>
constexpr static bool OrValue<false, TNext> = TNext::value;

// And
template <bool cur, typename TNext>
constexpr static bool AndValue = false;

template <typename TNext>
constexpr static bool AndValue<true, TNext> = TNext::value;

// Array empty
template <typename TArray>
struct IsArrayEmpty_;

template <template <typename...> class Cont, typename T1, typename...T2>
struct IsArrayEmpty_<Cont<T1, T2...>>
{
    constexpr static bool value = false;
};

template <template <typename...> class Cont>
struct IsArrayEmpty_<Cont<>>
{
    constexpr static bool value = true;
};

template <typename TArray>
constexpr static bool IsArrayEmpty = IsArrayEmpty_<TArray>::value;

// Array size
template <typename TArray>
struct ArraySize_;

template <template <typename...> class Cont, typename...T>
struct ArraySize_<Cont<T...>>
{
    constexpr static size_t value = sizeof...(T);
};

template <typename TArray>
constexpr static size_t ArraySize = ArraySize_<TArray>::value;

template <typename TSeqCont>
struct SeqHead_;

template <template <typename...> class Container, typename TH, typename...TCases>
struct SeqHead_<Container<TH, TCases...>>
{
    using type = TH;
};

template <typename TSeqCont>
using SeqHead = typename SeqHead_<TSeqCont>::type;

template <typename TSeqCont>
struct SeqTail_;

template <template <typename...> class Container, typename TH, typename...TCases>
struct SeqTail_<Container<TH, TCases...>>
{
    using type = Container<TCases...>;
};

template <typename TSeqCont>
using SeqTail = typename SeqTail_<TSeqCont>::type;

template <typename T>
constexpr static bool DependencyFalse = false;

template <typename T>
using RemConstRef = std::remove_cv_t<std::remove_reference_t<T>>;
}