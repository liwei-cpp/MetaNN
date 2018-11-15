#pragma once

#include <type_traits>

namespace MetaNN
{
/// data types
namespace CategoryTags
{
    struct Invalid;
    
    struct Scalar;
    struct Matrix;
    struct ThreeDArray;

    template <typename> struct Batch;
    template <typename> struct Sequence;
    template <typename> struct BatchSequence;

    using BatchScalar = Batch<Scalar>;
    using BatchMatrix = Batch<Matrix>;
    using BatchThreeDArray = Batch<ThreeDArray>;

    using ScalarSequence = Sequence<Scalar>;
    using MatrixSequence = Sequence<Matrix>;
    using ThreeDArraySequence = Sequence<ThreeDArray>;
    
    using BatchScalarSequence = BatchSequence<Scalar>;
    using BatchMatrixSequence = BatchSequence<Matrix>;
    using BatchThreeDArraySequence = BatchSequence<ThreeDArray>;
}

template <typename T>
constexpr bool IsValidCategoryTag = std::is_same_v<T, CategoryTags::Scalar> ||
                                    std::is_same_v<T, CategoryTags::Matrix> ||
                                    std::is_same_v<T, CategoryTags::ThreeDArray> ||
                                    std::is_same_v<T, CategoryTags::BatchScalar> ||
                                    std::is_same_v<T, CategoryTags::BatchMatrix> ||
                                    std::is_same_v<T, CategoryTags::BatchThreeDArray> ||
                                    std::is_same_v<T, CategoryTags::ScalarSequence> ||
                                    std::is_same_v<T, CategoryTags::MatrixSequence> ||
                                    std::is_same_v<T, CategoryTags::ThreeDArraySequence> ||
                                    std::is_same_v<T, CategoryTags::BatchScalarSequence> ||
                                    std::is_same_v<T, CategoryTags::BatchMatrixSequence> ||
                                    std::is_same_v<T, CategoryTags::BatchThreeDArraySequence>;

template <typename T>
constexpr bool IsBatchSequenceategoryTag = false;

template <typename TCardinal>
constexpr bool IsBatchSequenceategoryTag<CategoryTags::BatchSequence<TCardinal>> = true;
/// device types
struct DeviceTags
{
    struct CPU;
};
}
