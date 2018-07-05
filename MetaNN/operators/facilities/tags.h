#pragma once

namespace MetaNN
{
struct UnaryOpTags
{
    struct Abs;
    struct Sigmoid;
    struct Sign;
    struct Tanh;
    struct Transpose;
    struct Collapse;
    struct VecSoftmax;
};

struct BinaryOpTags
{
    struct Add;
    struct Substract;
    struct ElementMul;
    struct Divide;
    struct Dot;
    struct NegativeLogLikelihood;
    struct SigmoidDerivative;
    struct TanhDerivative;
    struct VecSoftmaxDerivative;
};

struct TernaryOpTags
{
    struct Interpolate;
    struct NegativeLogLikelihoodDerivative;
};
}