#include <data_gen.h>
#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_asin_case1()
    {
        cout << "Test asin case 1 (scalar)\t";
        {
            Scalar<CheckElement, CheckDevice> ori(static_cast<CheckElement>(0.2718));
            auto op = Asin(ori);
            auto res = Evaluate(op);
            assert(fabs(res.Value() - 0.2752) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_asin_case2()
    {
        cout << "Test asin case 2 (matrix)\t";
        auto ori = GenTensor<CheckElement>(-1, static_cast<CheckElement>(0.01), 10, 7);
        auto op = Asin(ori);
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
                auto value = std::asin(ori(i, k));
                assert(fabs(res(i, k) - value) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
    
    void test_asin_case3()
    {
        cout << "Test asin case 3 (3d-array)\t";
        auto ori = GenTensor<CheckElement>(-1, static_cast<CheckElement>(0.01), 2, 10, 7);
        auto op = Asin(ori);
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
                    auto value = std::asin(ori(p, i, k));
                    assert(fabs(res(p, i, k) - value) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_asin_case4()
    {
        cout << "Test asin case 4 (batch scalar)\t";
        auto ori = GenTensor<CheckElement>(-1, static_cast<CheckElement>(0.1), 10);
        auto op = Asin(ori);
        static_assert(IsVector<decltype(op)>);
        assert(op.Shape()[0] == 10);
        
        auto res = Evaluate(op);
        static_assert(IsVector<decltype(res)>);
        assert(res.Shape()[0] == 10);
        
        for (size_t i = 0; i < 10; ++i)
        {
            auto value = std::asin(ori[i].Value());
            assert(fabs(res[i].Value() - value) < 0.001f);
        }
        cout << "done" << endl;
    }
}

namespace Test::Operation::Math
{
    void test_asin()
    {
        test_asin_case1();
        test_asin_case2();
        test_asin_case3();
        test_asin_case4();
    }
}

namespace
{
    void test_asin_grad_case1()
    {
        cout << "Test asin-grad case 1 (scalar)\t";
        {
            Scalar<CheckElement, CheckDevice> grad(1.5);
            Scalar<CheckElement, CheckDevice> ori(static_cast<CheckElement>(0.3348));
            auto op = AsinGrad(grad, ori);
            auto res = Evaluate(op);
            
            auto check = 1.5 / std::sqrt(1 - 0.3348 * 0.3348);
            assert(fabs(res.Value() - check) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_asin_grad_case2()
    {
        cout << "Test asin-grad case 2 (matrix)\t";
        auto grad = GenTensor<CheckElement>(0, 1, 10, 7);
        auto ori = GenTensor<CheckElement>(static_cast<CheckElement>(-0.5), static_cast<CheckElement>(0.01), 10, 7);
        auto op = AsinGrad(grad, ori);
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
                auto check = grad(i, k) / std::sqrt(1 - ori(i, k) * ori(i, k));
                assert(fabs(res(i, k) - check) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
    
    void test_asin_grad_case3()
    {
        cout << "Test asin-grad case 3 (3d-array)\t";
        auto grad = GenTensor<CheckElement>(0, 1, 2, 10, 7);
        auto ori = GenTensor<CheckElement>(static_cast<CheckElement>(-0.9), static_cast<CheckElement>(0.01), 2, 10, 7);
        auto op = AsinGrad(grad, ori);
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
                    auto check = grad(p, i, k) / std::sqrt(1 - ori(p, i, k) * ori(p, i, k));
                    assert(fabs(res(p, i, k) - check) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_asin_grad_case4()
    {
        cout << "Test asin-grad case 4 (batch scalar)\t";
        auto grad = GenTensor<CheckElement>(0, 1, 10);
        auto ori = GenTensor<CheckElement>(static_cast<CheckElement>(-0.9), static_cast<CheckElement>(0.1), 10);
        auto op = AsinGrad(grad, ori);
        static_assert(IsTensorWithDim<decltype(op), 1>);
        assert(op.Shape()[0] == 10);
        
        auto res = Evaluate(op);
        static_assert(IsTensorWithDim<decltype(res), 1>);
        assert(res.Shape()[0] == 10);
        
        for (size_t i = 0; i < 10; ++i)
        {
            auto check = grad[i].Value() / std::sqrt(1 - ori[i].Value() * ori[i].Value());
            assert(fabs(res[i].Value() - check) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_asin_grad_case5()
    {
        cout << "Test asin-grad case 5 (grad broadcast)\t";
        auto grad = GenTensor<CheckElement>(0, 1, 10, 7);
        auto ori = GenTensor<CheckElement>(static_cast<CheckElement>(-0.9), static_cast<CheckElement>(0.01), 2, 10, 7);
        auto op = AsinGrad(grad, ori);
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
                    auto check = grad(i, k) / std::sqrt(1 - ori(p, i, k) * ori(p, i, k));
                    assert(fabs(res(p, i, k) - check) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operation::Math
{
    void test_asin_grad()
    {
        test_asin_grad_case1();
        test_asin_grad_case2();
        test_asin_grad_case3();
        test_asin_grad_case4();
        test_asin_grad_case5();
    }
}