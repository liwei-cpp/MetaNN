#pragma once

namespace MetaNN::OpTags
{
    // Data transform
    struct Duplicate;
    struct Collapse;
    
    // Elementwise operator
    struct Abs;
    struct Acos;        struct AcosGrad;
    struct Add;
    struct Asin;        struct AsinGrad;
    struct Divide;
    struct DivideByNumber;
    struct Interpolate;
    struct Multiply;
    struct Sigmoid;     struct SigmoidGrad;
    struct Sign;
    struct Substract;
    struct Tanh;        struct TanhGrad;
    
    // Mutating operators
    struct Transpose;
    
    // BLAS operators
    struct Dot;
    
    // non-linear activation operators
    struct Softmax;     struct SoftmaxGrad;
    
namespace ConvRelated
{
    struct Conv2D;
}
}