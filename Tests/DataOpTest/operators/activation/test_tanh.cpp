#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_tanh_case1()
    {
        cout << "Test tanh case 1 (scalar)\t";
        {
            Scalar<CheckElement, CheckDevice> ori(1.3122);
            auto op = Tanh(ori);
            auto res = Evaluate(op);
            assert(fabs(res.Value() - 0.8648) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_tanh_case2()
    {
        cout << "Test tanh case 2 (matrix)\t";
        auto ori = GenTensor<CheckElement>(-1, 0.01, 10, 7);
        auto op = Tanh(ori);
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
                auto value = std::tanh(ori(i, k));
                assert(fabs(res(i, k) - value) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
    
    void test_tanh_case3()
    {
        cout << "Test tanh case 3 (3d-array)\t";
        auto ori = GenTensor<CheckElement>(-1, 0.01, 2, 10, 7);
        auto op = Tanh(ori);
        static_assert(IsThreeDArray<decltype(op)>);
        assert(op.Shape()[0] == 2);
        assert(op.Shape()[1] == 10);
        assert(op.Shape()[2] == 7);
        
        auto res = Evaluate(op);
        static_assert(IsThreeDArray<decltype(res)>);
        assert(res.Shape()[0] == 2);
        assert(res.Shape()[1] == 10);
        assert(res.Shape()[2] == 7);
        
        for (size_t p = 0; p < 2; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    auto value = std::tanh(ori(p, i, k));
                    assert(fabs(res(p, i, k) - value) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_tanh_case4()
    {
        cout << "Test tanh case 4 (batch scalar)\t";
        auto ori = GenTensor<CheckElement>(-1, 0.1, 10);
        auto op = Tanh(ori);
        static_assert(IsTensorWithDim<decltype(op), 1>);
        assert(op.Shape()[0] == 10);
        
        auto res = Evaluate(op);
        static_assert(IsTensorWithDim<decltype(res), 1>);
        assert(res.Shape()[0] == 10);
        
        for (size_t i = 0; i < 10; ++i)
        {
            auto value = std::tanh(ori[i].Value());
            assert(fabs(res[i].Value() - value) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_tanh_case5()
    {
        cout << "Test tanh case 5 (batch 3d-array)\t";
        auto ori = GenTensor<CheckElement>(0, 1, 2, 3, 10, 7);
        auto op = Tanh(ori);
        static_assert(IsTensorWithDim<decltype(op), 4>);
        assert(op.Shape()[0] == 2);
        assert(op.Shape()[1] == 3);
        assert(op.Shape()[2] == 10);
        assert(op.Shape()[3] == 7);
        
        auto res = Evaluate(op);
        static_assert(IsTensorWithDim<decltype(res), 4>);
        assert(res.Shape()[0] == 2);
        assert(res.Shape()[1] == 3);
        assert(res.Shape()[2] == 10);
        assert(res.Shape()[3] == 7);
        
        for (size_t b = 0; b < 2; ++b)
        {
            for (size_t p = 0; p < 3; ++p)
            {
                for (size_t i = 0; i < 10; ++i)
                {
                    for (size_t k = 0; k < 7; ++k)
                    {
                        auto value = tanh(ori[b](p, i, k));
                        assert(fabs(res[b](p, i, k) - value) < 0.001f);
                    }
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators::Activation
{
    void test_tanh()
    {
        test_tanh_case1();
        test_tanh_case2();
        test_tanh_case3();
        test_tanh_case4();
        test_tanh_case5();
    }
}

namespace
{
    void test_tanh_grad_case1()
    {
        cout << "Test tanh grad case 1 (scalar)\t";
        Scalar<CheckElement, CheckDevice> grad(3);
        Scalar<CheckElement, CheckDevice> inpu(9);
        auto op = TanhGrad(grad, inpu);
        static_assert(IsScalar<decltype(op)>);
        
        auto res = Evaluate(op);
        static_assert(IsScalar<decltype(res)>);
        auto value = 3 * (1 - 9 * 9);
        assert(fabs(res.Value() - value) < 0.001f);
        cout << "done" << endl;
    }
    
    void test_tanh_grad_case2()
    {
        cout << "Test tanh grad case 2 (matrix)\t";
        auto grad = GenTensor<CheckElement>(-100, 3, 10, 7);
        auto inpu = GenTensor<CheckElement>(1, 1.5, 10, 7);
        auto op = TanhGrad(grad, inpu);
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
                auto value = grad(i, k) * (1 - inpu(i, k) * inpu(i, k));
                assert(fabs(value - res(i, k)) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators::Activation
{
    void test_tanh_grad()
    {
        test_tanh_grad_case1();
        test_tanh_grad_case2();
    }
}