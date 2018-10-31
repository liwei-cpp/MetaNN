#pragma once
#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/facilities/traits.h>
#include <type_traits>

namespace MetaNN
{
template <typename TElem, typename TDevice> class Matrix;
template <typename TElem, typename TDevice> class Scalar;
template <typename TElem, typename TDevice> class ThreeDArray;

template<typename TElem, typename TDevice, typename TCategory> class LinearTable;

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
    using type = LinearTable<TElem, TDevice, CategoryTags::BatchMatrix>;
};

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::BatchScalar, TElem, TDevice>
{
    using type = LinearTable<TElem, TDevice, CategoryTags::BatchScalar>;
};

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::BatchThreeDArray, TElem, TDevice>
{
    using type = LinearTable<TElem, TDevice, CategoryTags::BatchThreeDArray>;
};

template <typename TCategory, typename TElem, typename TDevice>
using PrincipalDataType = typename PrincipalDataType_<TCategory, TElem, TDevice>::type;

template <typename T>
struct DataCategory_
{
    using type = CategoryTags::Invalid;
};

template <typename T>
struct DataCategory_<const T>
{
    using type = typename DataCategory_<T>::type;
};

template <typename T>
struct DataCategory_<T&>
{
    using type = typename DataCategory_<T>::type;
};

template <typename T>
struct DataCategory_<const T&>
{
    using type = typename DataCategory_<T>::type;
};

template <typename T>
struct DataCategory_<T&&>
{
    using type = typename DataCategory_<T>::type;
};

template <typename T>
struct DataCategory_<const T&&>
{
    using type = typename DataCategory_<T>::type;
};

template <typename T>
using DataCategory = typename DataCategory_<T>::type;

template <typename T>
constexpr bool IsScalar = std::is_same_v<DataCategory<T>, CategoryTags::Scalar>;

template <typename T>
constexpr bool IsMatrix = std::is_same_v<DataCategory<T>, CategoryTags::Matrix>;

template <typename T>
constexpr bool IsThreeDArray = std::is_same_v<DataCategory<T>, CategoryTags::ThreeDArray>;

template <typename T>
constexpr bool IsBatchMatrix = std::is_same_v<DataCategory<T>, CategoryTags::BatchMatrix>;

template <typename T>
constexpr bool IsBatchScalar = std::is_same_v<DataCategory<T>, CategoryTags::BatchScalar>;

template <typename T>
constexpr bool IsBatchThreeDArray = std::is_same_v<DataCategory<T>, CategoryTags::BatchThreeDArray>;

template <typename T>
constexpr bool IsMatrixSequence = std::is_same_v<DataCategory<T>, CategoryTags::MatrixSequence>;

template <typename T>
constexpr bool IsScalarSequence = std::is_same_v<DataCategory<T>, CategoryTags::ScalarSequence>;

template <typename T>
constexpr bool IsThreeDArraySequence = std::is_same_v<DataCategory<T>, CategoryTags::ThreeDArraySequence>;

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
