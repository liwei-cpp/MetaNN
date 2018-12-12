#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_sigmoid_grad_case1()
    {
        cout << "Test sigmoid grad case 1 (scalar)\t";
        Scalar<CheckElement, CheckDevice> grad(3);
        Scalar<CheckElement, CheckDevice> inpu(9);
        auto op = SigmoidGrad(grad, inpu);
        static_assert(IsScalar<decltype(op)>);
        
        auto res = Evaluate(op);
        static_assert(IsScalar<decltype(res)>);
        auto value = 3 * 9 * (1 - 9);
        assert(fabs(res.Value() - value) < 0.001f);
        cout << "done" << endl;
    }
    
    void test_sigmoid_grad_case2()
    {
        cout << "Test sigmoid grad case 2 (matrix)\t";
        auto grad = GenMatrix<CheckElement>(10, 7, -100, 3);
        auto inpu = GenMatrix<CheckElement>(10, 7, 1, 1.5);
        auto op = SigmoidGrad(grad, inpu);
        static_assert(IsMatrix<decltype(op)>);
        assert(op.Shape().RowNum() == 10);
        assert(op.Shape().ColNum() == 7);
        
        auto res = Evaluate(op);
        static_assert(IsMatrix<decltype(res)>);
        assert(res.Shape().RowNum() == 10);
        assert(res.Shape().ColNum() == 7);
        
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                auto value = grad(i, k) * inpu(i, k) * (1 - inpu(i, k));
                assert(fabs(value - res(i, k)) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators
{
    void test_sigmoid_grad()
    {
        test_sigmoid_grad_case1();
        test_sigmoid_grad_case2();
    }
}