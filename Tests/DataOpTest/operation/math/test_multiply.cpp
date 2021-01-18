#include <data_gen.h>
#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_multiply_case1()
    {
        cout << "Test multiply case 1 (scalar)\t";
        Scalar<CheckElement, CheckDevice> ori1(3);
        Scalar<CheckElement, CheckDevice> ori2(9);
        auto op = ori1 * ori2;
        static_assert(IsScalar<decltype(op)>);
        
        auto res = Evaluate(op);
        static_assert(IsScalar<decltype(res)>);
        assert(fabs(res.Value() - 27) < 0.001f);
        cout << "done" << endl;
    }
    
    void test_multiply_case2()
    {
        cout << "Test multiply case 2 (matrix)\t";
        auto ori1 = GenTensor<CheckElement>(-100, 3, 10, 7);
        auto ori2 = GenTensor<CheckElement>(1, 1.5, 10, 7);
        auto op = ori1 * ori2;
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
                auto check = ori1(i, k) * ori2(i, k);
                assert(fabs(check - res(i, k)) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
    
    void test_multiply_case3()
    {
        cout << "Test multiply case 3 (3d-array)\t";
        auto ori1 = GenTensor<CheckElement>(-100, 3, 6, 10, 7);
        auto ori2 = GenTensor<CheckElement>(1, 1.5, 6, 10, 7);
        auto op = ori1 * ori2;
        static_assert(IsThreeDArray<decltype(op)>);
        assert(op.Shape()[0] == 6);
        assert(op.Shape()[1] == 10);
        assert(op.Shape()[2] == 7);
        
        auto res = Evaluate(op);
        static_assert(IsThreeDArray<decltype(res)>);
        assert(res.Shape()[0] == 6);
        assert(res.Shape()[1] == 10);
        assert(res.Shape()[2] == 7);
        
        for (size_t p = 0; p < 6; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    auto check = ori1(p, i, k) * ori2(p, i, k);
                    assert(fabs(check - res(p, i, k)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_multiply_case4()
    {
        cout << "Test multiply case 4 (batch scalar)\t";
        auto ori1 = GenTensor<CheckElement>(-100, 3, 6);
        auto ori2 = GenTensor<CheckElement>(1, 1.5, 6);
        auto op = ori1 * ori2;
        static_assert(IsVector<decltype(op)>);
        assert(op.Shape()[0] == 6);
        
        auto res = Evaluate(op);
        static_assert(IsVector<decltype(res)>);
        assert(res.Shape()[0] == 6);
        
        for (size_t p = 0; p < 6; ++p)
        {
            auto check = ori1[p].Value() * ori2[p].Value();
            assert(fabs(check - res[p].Value()) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_multiply_case5()
    {
        cout << "Test multiply case 3 (multiply with number)\t";
        {
            auto ori1 = GenTensor<CheckElement>(-100, 3, 10, 7);
            auto op = ori1 * static_cast<CheckElement>(3);
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
                    auto check = ori1(i, k) * 3;
                    assert(fabs(check - res(i, k)) < 0.001f);
                }
            }
        }
        {
            auto ori1 = GenTensor<CheckElement>(-100, 3, 10, 7);
            auto op = static_cast<CheckElement>(3) * ori1;
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
                    auto check = ori1(i, k) * 3;
                    assert(fabs(check - res(i, k)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_multiply_case6()
    {
        cout << "Test multiply case 6 (broadcast)\t";
        auto ori1 = GenTensor<CheckElement>(-100, 3, 10, 7);
        auto ori2 = GenTensor<CheckElement>(1, 1.5, 6, 10, 7);
        auto op = ori1 * ori2;
        static_assert(IsThreeDArray<decltype(op)>);
        assert(op.Shape()[0] == 6);
        assert(op.Shape()[1] == 10);
        assert(op.Shape()[2] == 7);
        
        auto res = Evaluate(op);
        static_assert(IsThreeDArray<decltype(res)>);
        assert(res.Shape()[0] == 6);
        assert(res.Shape()[1] == 10);
        assert(res.Shape()[2] == 7);
        
        for (size_t p = 0; p < 6; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    auto check = ori1(i, k) * ori2(p, i, k);
                    assert(fabs(check - res(p, i, k)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operation::Math
{
    void test_multiply()
    {
        test_multiply_case1();
        test_multiply_case2();
        test_multiply_case3();
        test_multiply_case4();
        test_multiply_case5();
        test_multiply_case6();
    }
}