#include <data_gen.h>
#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_acos_case1()
    {
        cout << "Test acos case 1 (scalar)\t";
        {
            Scalar<CheckElement, CheckDevice> ori(0.3348);
            auto op = Acos(ori);
            auto res = Evaluate(op);
            assert(fabs(res.Value() - 1.2294) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_acos_case2()
    {
        cout << "Test acos case 2 (matrix)\t";
        auto ori = GenTensor<CheckElement>(-1, 0.01, 10, 7);
        auto op = Acos(ori);
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
                auto value = std::acos(ori(i, k));
                assert(fabs(res(i, k) - value) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
    
    void test_acos_case3()
    {
        cout << "Test acos case 3 (3d-array)\t";
        auto ori = GenTensor<CheckElement>(-1, 0.01, 2, 10, 7);
        auto op = Acos(ori);
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
                    auto value = std::acos(ori(p, i, k));
                    assert(fabs(res(p, i, k) - value) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_acos_case4()
    {
        cout << "Test acos case 4 (batch scalar)\t";
        auto ori = GenTensor<CheckElement>(-1, 0.1, 10);
        auto op = Acos(ori);
        static_assert(IsTensorWithDim<decltype(op), 1>);
        assert(op.Shape()[0] == 10);
        
        auto res = Evaluate(op);
        static_assert(IsTensorWithDim<decltype(res), 1>);
        assert(res.Shape()[0] == 10);
        
        for (size_t i = 0; i < 10; ++i)
        {
            auto value = std::acos(ori[i].Value());
            assert(fabs(res[i].Value() - value) < 0.001f);
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators::Elwentwise
{
    void test_acos()
    {
        test_acos_case1();
        test_acos_case2();
        test_acos_case3();
        test_acos_case4();
    }
}

namespace
{
    void test_acos_grad_case1()
    {
        cout << "Test acos-grad case 1 (scalar)\t";
        {
            Scalar<CheckElement, CheckDevice> grad(1.5);
            Scalar<CheckElement, CheckDevice> ori(0.3348);
            auto op = AcosGrad(grad, ori);
            auto res = Evaluate(op);
            
            auto check = -1.5 / std::sqrt(1 - 0.3348 * 0.3348);
            assert(fabs(res.Value() - check) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_acos_grad_case2()
    {
        cout << "Test acos-grad case 2 (matrix)\t";
        auto grad = GenTensor<CheckElement>(0, 1, 10, 7);
        auto ori = GenTensor<CheckElement>(-0.5, 0.01, 10, 7);
        auto op = AcosGrad(grad, ori);
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
                auto check = -grad(i, k) / std::sqrt(1 - ori(i, k) * ori(i, k));
                assert(fabs(res(i, k) - check) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
    
    void test_acos_grad_case3()
    {
        cout << "Test acos-grad case 3 (3d-array)\t";
        auto grad = GenTensor<CheckElement>(0, 1, 2, 10, 7);
        auto ori = GenTensor<CheckElement>(-0.9, 0.01, 2, 10, 7);
        auto op = AcosGrad(grad, ori);
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
                    auto check = -grad(p, i, k) / std::sqrt(1 - ori(p, i, k) * ori(p, i, k));
                    assert(fabs(res(p, i, k) - check) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_acos_grad_case4()
    {
        cout << "Test acos-grad case 4 (batch scalar)\t";
        auto grad = GenTensor<CheckElement>(0, 1, 10);
        auto ori = GenTensor<CheckElement>(-0.9, 0.1, 10);
        auto op = AcosGrad(grad, ori);
        static_assert(IsTensorWithDim<decltype(op), 1>);
        assert(op.Shape()[0] == 10);
        
        auto res = Evaluate(op);
        static_assert(IsTensorWithDim<decltype(res), 1>);
        assert(res.Shape()[0] == 10);
        
        for (size_t i = 0; i < 10; ++i)
        {
            auto check = -grad[i].Value() / std::sqrt(1 - ori[i].Value() * ori[i].Value());
            assert(fabs(res[i].Value() - check) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_acos_grad_case5()
    {
        cout << "Test acos-grad case 5 (grad broadcast)\t";
        auto grad = GenTensor<CheckElement>(0, 1, 10, 7);
        auto ori = GenTensor<CheckElement>(-0.9, 0.01, 2, 10, 7);
        auto op = AcosGrad(grad, ori);
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
                    auto check = -grad(i, k) / std::sqrt(1 - ori(p, i, k) * ori(p, i, k));
                    assert(fabs(res(p, i, k) - check) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators::Elwentwise
{
    void test_acos_grad()
    {
        test_acos_grad_case1();
        test_acos_grad_case2();
        test_acos_grad_case3();
        test_acos_grad_case4();
        test_acos_grad_case5();
    }
}