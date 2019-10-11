#pragma once
#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/facilities/traits.h>
#include <MetaNN/facilities/null_param.h>
#include <iterator>
#include <type_traits>

namespace MetaNN
{
template <typename TElem, typename TDevice> class Matrix;
template <typename TElem, typename TDevice> class Scalar;
template <typename TElem, typename TDevice> class ThreeDArray;

template <typename TElement, typename TDevice,
          template<typename>class TCateWrapper, typename TCardinalCate>
class StaticArray;

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
    using type = StaticArray<TElem, TDevice, CategoryTags::Batch, CategoryTags::Matrix>;
};

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::BatchScalar, TElem, TDevice>
{
    using type = StaticArray<TElem, TDevice, CategoryTags::Batch, CategoryTags::Scalar>;
};

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::BatchThreeDArray, TElem, TDevice>
{
    using type = StaticArray<TElem, TDevice, CategoryTags::Batch, CategoryTags::ThreeDArray>;
};

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::MatrixSequence, TElem, TDevice>
{
    using type = StaticArray<TElem, TDevice, CategoryTags::Sequence, CategoryTags::Matrix>;
};

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::ScalarSequence, TElem, TDevice>
{
    using type = StaticArray<TElem, TDevice, CategoryTags::Sequence, CategoryTags::Scalar>;
};

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::ThreeDArraySequence, TElem, TDevice>
{
    using type = StaticArray<TElem, TDevice, CategoryTags::Sequence, CategoryTags::ThreeDArray>;
};

// batch sequence
template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::BatchMatrixSequence, TElem, TDevice>
{
    using type = StaticArray<TElem, TDevice, CategoryTags::BatchSequence, CategoryTags::Matrix>;
};

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::BatchScalarSequence, TElem, TDevice>
{
    using type = StaticArray<TElem, TDevice, CategoryTags::BatchSequence, CategoryTags::Scalar>;
};

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::BatchThreeDArraySequence, TElem, TDevice>
{
    using type = StaticArray<TElem, TDevice, CategoryTags::BatchSequence, CategoryTags::ThreeDArray>;
};

template <typename TCategory, typename TElem, typename TDevice>
using PrincipalDataType = typename PrincipalDataType_<TCategory, TElem, TDevice>::type;

template <typename T>
struct HasCategoryTag_
{   
};

template <typename T>
constexpr bool HasCategoryTag = HasCategoryTag_<T>::value;

template <typename T>
struct DataCategory_
{
    template <typename R>
    static typename R::CategoryTag Test(typename R::CategoryTag*);
    
    template <typename R>
    static CategoryTags::OutOfCategory Test(...);

    using typeCand = decltype(Test<T>(nullptr));
    using type = std::conditional_t<IsValidCategoryTag<typeCand>,
                                    typeCand,
                                    CategoryTags::OutOfCategory>;
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
constexpr bool IsOutOfDataCategory = std::is_same_v<DataCategory<T>, CategoryTags::OutOfCategory>;

template <typename T>
constexpr bool IsInDataCategory = !IsOutOfDataCategory<T>;

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
constexpr bool IsBatchMatrixSequence = std::is_same_v<DataCategory<T>, CategoryTags::BatchMatrixSequence>;

template <typename T>
constexpr bool IsBatchScalarSequence = std::is_same_v<DataCategory<T>, CategoryTags::BatchScalarSequence>;

template <typename T>
constexpr bool IsBatchThreeDArraySequence = std::is_same_v<DataCategory<T>, CategoryTags::BatchThreeDArraySequence>;

template <typename T>
constexpr bool IsCardinal = IsScalar<T> || IsMatrix<T> || IsThreeDArray<T>;

template <typename T>
constexpr bool IsBatchCardinal = IsBatchScalar<T> || IsBatchMatrix<T> || IsBatchThreeDArray<T>;

template <typename T>
constexpr bool IsCardinalSequence = IsScalarSequence<T> || IsMatrixSequence<T> || IsThreeDArraySequence<T>;

template <typename T>
constexpr bool IsBatchCardinalSequence = IsBatchScalarSequence<T> || IsBatchMatrixSequence<T> || IsBatchThreeDArraySequence<T>;

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

template <typename T1, typename T2,
          typename = std::enable_if_t<IsInDataCategory<T1>>,
          typename = std::enable_if_t<IsInDataCategory<T2>>,
          typename = std::enable_if_t<!std::is_same_v<T1, T2>>
         >
bool operator== (const T1&, const T2&)
{
    return false;
}

template <typename T1, typename T2,
          typename = std::enable_if_t<IsInDataCategory<T1>>,
          typename = std::enable_if_t<IsInDataCategory<T2>>
         >
bool operator!= (const T1& val1, const T2& val2)
{
    return !(val1 == val2);
}

template <typename TData>
using ShapeType = RemConstRef<decltype(std::declval<TData>().Shape())>;
}
