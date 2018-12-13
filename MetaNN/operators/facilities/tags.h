#pragma once

namespace MetaNN::OpTags
{
    // Data transform
    struct Duplicate;
    struct Collapse;
    
    // Elementwise operator
    struct Abs;
    struct Acos;    struct AcosGrad;
    struct Add;
    struct Asin;    struct AsinGrad;
    struct Divide;
    struct Multiply;
    struct Sigmoid; struct SigmoidGrad;
    struct Sign;
    struct Substract;
    struct Tanh;
    
namespace UnaryOpTags
{
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