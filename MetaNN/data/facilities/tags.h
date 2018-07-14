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
    template <typename> struct Seq;

    using BatchScalar = Batch<Scalar>;
    using BatchMatrix = Batch<Matrix>;
    using BatchThreeDArray = Batch<ThreeDArray>;

    using SeqScalar = Seq<Scalar>;
    using SeqMatrix = Seq<Matrix>;
    using SeqThreeDArray = Seq<ThreeDArray>;
}

/// device types
struct DeviceTags
{
    struct CPU;
};
}
