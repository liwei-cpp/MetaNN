#pragma once

namespace MetaNN::OpTags
{
    // Data transform
    struct Duplicate;
    struct Collapse;
    
    // Elementwise operator
    struct Abs;
    struct Add;
    struct Divide;
    struct Multiply;
    struct Sign;
    struct Substract;
    
namespace UnaryOpTags
{
    struct Sigmoid;
    struct Tanh;
    struct Transpose;
    struct Collapse;
    struct VecSoftmax;
};

namespace BinaryOpTags
{
    struct Dot;
    struct NegativeLogLikelihood;
    struct SigmoidDerivative;
    struct TanhDerivative;
    struct VecSoftmaxDerivative;
};

namespace TernaryOpTags
{
    struct Interpolate;
    struct NegativeLogLikelihoodDerivative;
};

namespace ConvRelated
{
    struct Conv2D;
}
}