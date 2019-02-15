#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_relu_case1()
    {
        cout << "Test ReLU case 1\t";
        auto ori = GenMatrix<CheckElement>(10, 7, -10, 1);
        auto op = ReLU(ori);
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
                if (ori(i, k) > 0)
                {
                    assert(fabs(res(i, k) - ori(i, k)) < 0.001f);
                }
                else
                {
                    assert(fabs(res(i, k)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators
{
    void test_relu()
    {
        test_relu_case1();
    }
}

namespace
{
    void test_relu_grad_case1()
    {
        cout << "Test ReLU grad case 1\t";
        auto input = GenMatrix<CheckElement>(10, 7, -10, 1);
        auto grad = GenMatrix<CheckElement>(10, 7, 0, 0.1);
        auto op = ReLUGrad(grad, input);
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
                if (input(i, k) > 0)
                {
                    assert(fabs(res(i, k) - grad(i, k)) < 0.001f);
                }
                else
                {
                    assert(fabs(res(i, k)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators
{
    void test_relu_grad()
    {
        test_relu_grad_case1();
    }
}