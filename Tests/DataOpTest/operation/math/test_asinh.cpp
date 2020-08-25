#include <data_gen.h>
#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_asinh_case1()
    {
        cout << "Test asinh case 1 (scalar)\t";
        {
            Scalar<CheckElement, CheckDevice> ori(1.2f);
            auto op = Asinh(ori);
            auto res = Evaluate(op);
            assert(fabs(res.Value() - 1.0160) < 0.001f);
        }
        cout << "done" << endl;
    }

    void test_asinh_case2()
    {
        cout << "Test asinh case 2 (vector)\t";
        {
            Vector<CheckElement, CheckDevice> ori(6);
            ori.SetValue(0, -2.f);
            ori.SetValue(1, -0.5f);
            ori.SetValue(2, 1.f);
            ori.SetValue(3, 1.2f);
            ori.SetValue(4, 200);
            ori.SetValue(5, 10000);
            auto op = Asinh(ori);
            auto res = Evaluate(op);
            assert(fabs(res(0) + 1.4436) < 0.001f);
            assert(fabs(res(1) + 0.4812) < 0.001f);
            assert(fabs(res(2) - 0.8814) < 0.001f);
            assert(fabs(res(3) - 1.0160) < 0.001f);
            assert(fabs(res(4) - 5.9915) < 0.001f);
            assert(fabs(res(5) - 9.9035) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_asinh_case3()
    {
        cout << "Test asinh case 3 (matrix)\t";
        auto ori = GenTensor<CheckElement>(1.1, 0.01, 10, 7);
        auto op = Asinh(ori);
        static_assert(IsMatrix<decltype(op)>);
        assert(op.Shape()[0] == 10);
        assert(op.Shape()[1] == 7);
        
        auto res = Evaluate(op);
        static_assert(IsMatrix<decltype(res)>);
        assert(res.Shape()[0] == 10);
        assert(res.Shape()[1] == 7);
        
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                auto value = std::asinh(ori(i, k));
                assert(fabs(res(i, k) - value) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operation::Math
{
    void test_asinh()
    {
        test_asinh_case1();
        test_asinh_case2();
        test_asinh_case3();
    }
}

namespace
{
    void test_asinh_grad_case1()
    {
        cout << "Test asinh-grad case 1 (scalar)\t";
        {
            Scalar<CheckElement, CheckDevice> grad(1);
            Scalar<CheckElement, CheckDevice> ori(1.2);
            auto op = AsinhGrad(grad, ori);
            auto res = Evaluate(op);
            
            assert(fabs(res.Value() - 0.6402) < 0.001f);
        }
        cout << "done" << endl;
    }

    void test_asinh_grad_case2()
    {
        cout << "Test asinh-grad case 2 (vector)\t";
        {
            Scalar<CheckElement, CheckDevice> grad(1);
            Vector<CheckElement, CheckDevice> ori(6);
            ori.SetValue(0, -2.f);
            ori.SetValue(1, -0.5f);
            ori.SetValue(2, 1.f);
            ori.SetValue(3, 1.2f);
            ori.SetValue(4, 200);
            ori.SetValue(5, 10000);
            auto op = AsinhGrad(grad, ori);
            auto res = Evaluate(op);
            
            assert(fabs(res(0) - 0.4472) < 0.001f);
            assert(fabs(res(1) - 0.8944) < 0.001f);
            assert(fabs(res(2) - 0.7071) < 0.001f);
            assert(fabs(res(3) - 0.6402) < 0.001f);
            assert(fabs(res(4) - 0.0050) < 0.001f);
            assert(fabs(res(5) - 0.0001) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_asinh_grad_case3()
    {
        cout << "Test asinh-grad case 3 (matrix)\t";
        auto grad = GenTensor<CheckElement>(0, 1, 10, 7);
        auto ori = GenTensor<CheckElement>(1.5, 0.01, 10, 7);
        auto op = AsinhGrad(grad, ori);
        static_assert(IsMatrix<decltype(op)>);
        assert(op.Shape()[0] == 10);
        assert(op.Shape()[1] == 7);
        
        auto res = Evaluate(op);
        static_assert(IsMatrix<decltype(res)>);
        assert(res.Shape()[0] == 10);
        assert(res.Shape()[1] == 7);
        
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                auto check = grad(i, k) / std::sqrt(ori(i, k) * ori(i, k) + 1);
                assert(fabs(res(i, k) - check) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operation::Math
{
    void test_asinh_grad()
    {
        test_asinh_grad_case1();
        test_asinh_grad_case2();
        test_asinh_grad_case3();
    }
}