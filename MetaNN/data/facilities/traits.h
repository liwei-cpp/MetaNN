#pragma once
#include <MetaNN/data/facilities/tags.h>
#include <type_traits>

namespace MetaNN
{
template <typename TElem, typename TDevice> class Matrix;
template <typename TElem, typename TDevice> class Scalar;
template <typename TElem, typename TDevice> class ThreeDArray;

template<typename TElement, typename TDevice, typename TCategory> class Batch;

template <typename TCategory, typename TElem, typename TDevice>
struct PrincipalDataType_;

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::Scalar, TElem, TDevice>
{
    using type = Scalar<TElem, TDevice>;
};

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::Matrix, TElem, TDevice>
{
    using type = Matrix<TElem, TDevice>;
};

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::ThreeDArray, TElem, TDevice>
{
    using type = ThreeDArray<TElem, TDevice>;
};

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::BatchMatrix, TElem, TDevice>
{
    using type = Batch<TElem, TDevice, CategoryTags::Matrix>;
};

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::BatchScalar, TElem, TDevice>
{
    using type = Batch<TElem, TDevice, CategoryTags::Scalar>;
};

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::BatchThreeDArray, TElem, TDevice>
{
    using type = Batch<TElem, TDevice, CategoryTags::ThreeDArray>;
};

template <typename TCategory, typename TElem, typename TDevice>
using PrincipalDataType = typename PrincipalDataType_<TCategory, TElem, TDevice>::type;

/// is scalar
template <typename T>
constexpr bool IsScalar = false;

template <typename T>
constexpr bool IsScalar<const T> = IsScalar<T>;

template <typename T>
constexpr bool IsScalar<T&> = IsScalar<T>;

template <typename T>
constexpr bool IsScalar<T&&> = IsScalar<T>;

/// is matrix
template <typename T>
constexpr bool IsMatrix = false;

template <typename T>
constexpr bool IsMatrix<const T> = IsMatrix<T>;

template <typename T>
constexpr bool IsMatrix<T&> = IsMatrix<T>;

template <typename T>
constexpr bool IsMatrix<T&&> = IsMatrix<T>;

/// is batch scalar
template <typename T>
constexpr bool IsBatchScalar = false;

template <typename T>
constexpr bool IsBatchScalar<const T> = IsBatchScalar<T>;

template <typename T>
constexpr bool IsBatchScalar<T&> = IsBatchScalar<T>;

template <typename T>
constexpr bool IsBatchScalar<const T&> = IsBatchScalar<T>;

template <typename T>
constexpr bool IsBatchScalar<T&&> = IsBatchScalar<T>;

template <typename T>
constexpr bool IsBatchScalar<const T&&> = IsBatchScalar<T>;

/// is batch matrix
template <typename T>
constexpr bool IsBatchMatrix = false;

template <typename T>
constexpr bool IsBatchMatrix<const T> = IsBatchMatrix<T>;

template <typename T>
constexpr bool IsBatchMatrix<T&> = IsBatchMatrix<T>;

template <typename T>
constexpr bool IsBatchMatrix<const T&> = IsBatchMatrix<T>;

template <typename T>
constexpr bool IsBatchMatrix<T&&> = IsBatchMatrix<T>;

template <typename T>
constexpr bool IsBatchMatrix<const T&&> = IsBatchMatrix<T>;

/// is 3d array
template <typename T>
constexpr bool IsThreeDArray = false;

template <typename T>
constexpr bool IsThreeDArray<const T> = IsThreeDArray<T>;

template <typename T>
constexpr bool IsThreeDArray<T&> = IsThreeDArray<T>;

template <typename T>
constexpr bool IsThreeDArray<const T&> = IsThreeDArray<T>;

template <typename T>
constexpr bool IsThreeDArray<T&&> = IsThreeDArray<T>;

template <typename T>
constexpr bool IsThreeDArray<const T&&> = IsThreeDArray<T>;

/// is batch matrix
template <typename T>
constexpr bool IsBatchThreeDArray = false;

template <typename T>
constexpr bool IsBatchThreeDArray<const T> = IsBatchThreeDArray<T>;

template <typename T>
constexpr bool IsBatchThreeDArray<T&> = IsBatchThreeDArray<T>;

template <typename T>
constexpr bool IsBatchThreeDArray<const T&> = IsBatchThreeDArray<T>;

template <typename T>
constexpr bool IsBatchThreeDArray<T&&> = IsBatchThreeDArray<T>;

template <typename T>
constexpr bool IsBatchThreeDArray<const T&&> = IsBatchThreeDArray<T>;

namespace NSDataCategory
{
using tt = std::true_type*;
using ft = std::false_type*;
CategoryTags::Scalar      apply(tt, ft...);
CategoryTags::Matrix      apply(ft, tt, ft...);
CategoryTags::BatchScalar apply(ft, ft, tt, ft...);
CategoryTags::BatchMatrix apply(ft, ft, ft, tt, ft...);
CategoryTags::ThreeDArray apply(ft, ft, ft, ft, tt, ft...);
CategoryTags::BatchThreeDArray apply(ft, ft, ft, ft, ft, tt, ft...);
};

template <typename T>
using DataCategory = decltype(NSDataCategory::apply(
                              std::conditional_t<IsScalar<T>,      std::true_type*, std::false_type*>(),
                              std::conditional_t<IsMatrix<T>,      std::true_type*, std::false_type*>(),
                              std::conditional_t<IsBatchScalar<T>, std::true_type*, std::false_type*>(),
                              std::conditional_t<IsBatchMatrix<T>, std::true_type*, std::false_type*>(),
                              std::conditional_t<IsThreeDArray<T>, std::true_type*, std::false_type*>(),
                              std::conditional_t<IsBatchThreeDArray<T>, std::true_type*, std::false_type*>(),
                              (std::false_type*){}
                              ));

template <typename T>
struct IsIterator_
{   
    template <typename R>
    static std::true_type Test(typename std::iterator_traits<R>::iterator_category*);
    
    template <typename R>
    static std::false_type Test(...);

    static constexpr bool value = decltype(Test<T>(nullptr))::value;
};

template <typename T>
constexpr bool IsIterator = IsIterator_<T>::value;
}
