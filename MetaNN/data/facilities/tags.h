#pragma once

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

    using BatchScalar = Batch<Scalar>;
    using BatchMatrix = Batch<Matrix>;
    using BatchThreeDArray = Batch<ThreeDArray>;

    using ScalarSequence = Sequence<Scalar>;
    using MatrixSequence = Sequence<Matrix>;
    using ThreeDArraySequence = Sequence<ThreeDArray>;
}

/// device types
struct DeviceTags
{
    struct CPU;
};
}
