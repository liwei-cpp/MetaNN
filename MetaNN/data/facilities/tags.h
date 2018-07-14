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
    template <typename> struct BatchSeq;

    using BatchScalar = Batch<Scalar>;
    using BatchMatrix = Batch<Matrix>;
    using BatchThreeDArray = Batch<ThreeDArray>;

    using SeqScalar = Seq<Scalar>;
    using SeqMatrix = Seq<Matrix>;
    using SeqThreeDArray = Seq<ThreeDArray>;

    using BatchSeqScalar = BatchSeq<Scalar>;
    using BatchSeqMatrix = BatchSeq<Matrix>;
    using BatchSeqThreeDArray = BatchSeq<ThreeDArray>;
}

/// device types
struct DeviceTags
{
    struct CPU;
};
}
