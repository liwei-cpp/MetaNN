#pragma once

namespace MetaNN::OpTags
{
    // Data transform
    struct Duplicate;
    struct Collapse;
    
    // Element-wise operator
    struct Abs;
    
namespace UnaryOpTags
{
    struct Sigmoid;
    struct Sign;
    struct Tanh;
    struct Transpose;
    struct Collapse;
    struct VecSoftmax;
};

namespace BinaryOpTags
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