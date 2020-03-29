#include <data_gen.h>
#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_substract_case1()
    {
        cout << "Test substract case 1 (scalar)\t";
        Scalar<CheckElement, CheckDevice> ori1(3);
        Scalar<CheckElement, CheckDevice> ori2(9);
        auto op = ori1 - ori2;
        static_assert(IsScalar<decltype(op)>);
        
        auto res = Evaluate(op);
        static_assert(IsScalar<decltype(res)>);
        assert(fabs(res.Value() + 6) < 0.001f);
        cout << "done" << endl;
    }
    
    void test_substract_case2()
    {
        cout << "Test substract case 2 (matrix)\t";
        auto ori1 = GenTensor<CheckElement>(-100, 3, 10, 7);
        auto ori2 = GenTensor<CheckElement>(1, 1.5, 10, 7);
        auto op = ori1 - ori2;
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
                auto check = ori1(i, k) - ori2(i, k);
                assert(fabs(check - res(i, k)) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
    
    void test_substract_case3()
    {
        cout << "Test substract case 3 (3d-array)\t";
        auto ori1 = GenTensor<CheckElement>(-100, 3, 6, 10, 7);
        auto ori2 = GenTensor<CheckElement>(1, 1.5, 6, 10, 7);
        auto op = ori1 - ori2;
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
                    auto check = ori1(p, i, k) - ori2(p, i, k);
                    assert(fabs(check - res(p, i, k)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_substract_case4()
    {
        cout << "Test substract case 4\t";
        auto ori = GenTensor<CheckElement>(-100, 3, 6, 10, 7);
        auto op = 1 - ori;
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
                    auto check = 1 - ori(p, i, k);
                    assert(fabs(check - res(p, i, k)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_substract_case5()
    {
        cout << "Test substract case 5 (broadcast)\t";
        auto ori1 = GenTensor<CheckElement>(-100, 3, 6, 10, 7);
        auto ori2 = GenTensor<CheckElement>(1, 1.5, 10, 7);
        auto op = ori1 - ori2;
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
                    auto check = ori1(p, i, k) - ori2(i, k);
                    assert(fabs(check - res(p, i, k)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_substract_case6()
    {
        cout << "Test substract case 6\t";
        auto ori = GenTensor<CheckElement>(-100, 3, 6, 10, 7);
        auto op = ori - 1;
        static_assert(IsThreeDArray<decltype(op)>);
        assert(op.Shape() == Shape(6, 10, 7));
        
        auto res = Evaluate(op);
        static_assert(IsThreeDArray<decltype(res)>);
        assert(res.Shape() == Shape(6, 10, 7));
        
        for (size_t p = 0; p < 6; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    auto check = ori(p, i, k) - 1;
                    assert(fabs(check - res(p, i, k)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operation::Elwentwise
{
    void test_substract()
    {
        test_substract_case1();
        test_substract_case2();
        test_substract_case3();
        test_substract_case4();
        test_substract_case5();
        test_substract_case6();
    }
}