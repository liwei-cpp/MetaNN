#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
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
        auto ori = GenMatrix<CheckElement>(10, 7, -1, 0.01);
        auto op = Acos(ori);
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
                auto value = std::acos(ori(i, k));
                assert(fabs(res(i, k) - value) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
    
    void test_acos_case3()
    {
        cout << "Test acos case 3 (3d-array)\t";
        auto ori = GenThreeDArray<CheckElement>(2, 10, 7, -1, 0.01);
        auto op = Acos(ori);
        static_assert(IsThreeDArray<decltype(op)>);
        assert(op.Shape().PageNum() == 2);
        assert(op.Shape().RowNum() == 10);
        assert(op.Shape().ColNum() == 7);
        
        auto res = Evaluate(op);
        static_assert(IsThreeDArray<decltype(res)>);
        assert(res.Shape().PageNum() == 2);
        assert(res.Shape().RowNum() == 10);
        assert(res.Shape().ColNum() == 7);
        
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
        auto ori = GenBatchScalar<CheckElement>(10, -1, 0.1);
        auto op = Acos(ori);
        static_assert(IsBatchScalar<decltype(op)>);
        assert(op.Shape().BatchNum() == 10);
        
        auto res = Evaluate(op);
        static_assert(IsBatchScalar<decltype(res)>);
        assert(res.Shape().BatchNum() == 10);
        
        for (size_t i = 0; i < 10; ++i)
        {
            auto value = std::acos(ori[i].Value());
            assert(fabs(res[i].Value() - value) < 0.001f);
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators
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
        auto grad = GenMatrix<CheckElement>(10, 7);
        auto ori = GenMatrix<CheckElement>(10, 7, -0.5, 0.01);
        auto op = AcosGrad(grad, ori);
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
                auto check = -grad(i, k) / std::sqrt(1 - ori(i, k) * ori(i, k));
                assert(fabs(res(i, k) - check) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
    
    void test_acos_grad_case3()
    {
        cout << "Test acos-grad case 3 (3d-array)\t";
        auto grad = GenThreeDArray<CheckElement>(2, 10, 7);
        auto ori = GenThreeDArray<CheckElement>(2, 10, 7, -0.9, 0.01);
        auto op = AcosGrad(grad, ori);
        static_assert(IsThreeDArray<decltype(op)>);
        assert(op.Shape().PageNum() == 2);
        assert(op.Shape().RowNum() == 10);
        assert(op.Shape().ColNum() == 7);
        
        auto res = Evaluate(op);
        static_assert(IsThreeDArray<decltype(res)>);
        assert(res.Shape().PageNum() == 2);
        assert(res.Shape().RowNum() == 10);
        assert(res.Shape().ColNum() == 7);
        
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
        auto grad = GenBatchScalar<CheckElement>(10);
        auto ori = GenBatchScalar<CheckElement>(10, -0.9, 0.1);
        auto op = AcosGrad(grad, ori);
        static_assert(IsBatchScalar<decltype(op)>);
        assert(op.Shape().BatchNum() == 10);
        
        auto res = Evaluate(op);
        static_assert(IsBatchScalar<decltype(res)>);
        assert(res.Shape().BatchNum() == 10);
        
        for (size_t i = 0; i < 10; ++i)
        {
            auto check = -grad[i].Value() / std::sqrt(1 - ori[i].Value() * ori[i].Value());
            assert(fabs(res[i].Value() - check) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_acos_grad_case5()
    {
        cout << "Test acos-grad case 5 (batch matrix)\t";
        auto grad = GenBatchMatrix<CheckElement>(2, 10, 7);
        auto ori = GenBatchMatrix<CheckElement>(2, 10, 7, -0.9, 0.01);
        auto op = AcosGrad(grad, ori);
        static_assert(IsBatchMatrix<decltype(op)>);
        assert(op.Shape().BatchNum() == 2);
        assert(op.Shape().RowNum() == 10);
        assert(op.Shape().ColNum() == 7);
        
        auto res = Evaluate(op);
        static_assert(IsBatchMatrix<decltype(res)>);
        assert(res.Shape().BatchNum() == 2);
        assert(res.Shape().RowNum() == 10);
        assert(res.Shape().ColNum() == 7);
        
        for (size_t p = 0; p < 2; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    auto check = -grad[p](i, k) / std::sqrt(1 - ori[p](i, k) * ori[p](i, k));
                    assert(fabs(res[p](i, k) - check) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators
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