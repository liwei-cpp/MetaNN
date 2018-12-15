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
    struct Interpolate;
    struct Multiply;
    struct Sigmoid; struct SigmoidGrad;
    struct Sign;
    struct Substract;
    struct Tanh;    struct TanhGrad;
    
    // Mutating operators
    struct Transpose;
    
    struct Dot;
    
namespace UnaryOpTags
{
    struct VecSoftmax;
};

namespace BinaryOpTags
{
    struct NegativeLogLikelihood;
    struct SigmoidDerivative;
    struct VecSoftmaxDerivative;
};

namespace TernaryOpTags
{
    struct NegativeLogLikelihoodDerivative;
};

namespace ConvRelated
{
    struct Conv2D;
}
}