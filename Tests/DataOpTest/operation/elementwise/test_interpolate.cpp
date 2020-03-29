#include <data_gen.h>
#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_interpolate_case1()
    {
        cout << "Test interpolate case 1 (scalar)\t";
        Scalar<CheckElement, CheckDevice> ori1(3);
        Scalar<CheckElement, CheckDevice> ori2(9);
        Scalar<CheckElement, CheckDevice> lambda(0.3);
        auto op = Interpolate(ori1, ori2, lambda);
        static_assert(IsScalar<decltype(op)>);
        
        auto res = Evaluate(op);
        static_assert(IsScalar<decltype(res)>);
        CheckElement value = 3 * 0.3 + 9 * (1 - 0.3);
        assert(fabs(res.Value() - value) < 0.001f);
        cout << "done" << endl;
    }
    
    void test_interpolate_case2()
    {
        cout << "Test interpolate case 2 (matrix)\t";
        auto ori1 = GenTensor<CheckElement>(-100, 3, 10, 7);
        auto ori2 = GenTensor<CheckElement>(1, 1.5, 10, 7);
        auto lambda = GenTensor<CheckElement>(0.1, -2, 10, 7);
        auto op = Interpolate(ori1, ori2, lambda);
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
                auto check = ori1(i, k) * lambda(i, k) + ori2(i, k) * (1 - lambda(i, k));
                assert(fabs(check - res(i, k)) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
    
    void test_interpolate_case3()
    {
        cout << "Test interpolate case 3 (broadcast)\t";
        auto ori1 = GenTensor<CheckElement>(-100, 3, 10, 7);
        auto ori2 = GenTensor<CheckElement>(1, 1.5, 5, 10, 7);
        auto lambda = GenTensor<CheckElement>(0.1, -2, 7);
        auto op = Interpolate(ori1, ori2, lambda);
        static_assert(IsThreeDArray<decltype(op)>);
        assert(op.Shape()[0] == 5);
        assert(op.Shape()[1] == 10);
        assert(op.Shape()[2] == 7);
        
        auto res = Evaluate(op);
        static_assert(IsThreeDArray<decltype(res)>);
        assert(res.Shape()[0] == 5);
        assert(res.Shape()[1] == 10);
        assert(res.Shape()[2] == 7);
        
        for (size_t p = 0; p < 5; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    auto check = ori1(i, k) * lambda(k) + ori2(p, i, k) * (1 - lambda(k));
                    assert(fabs(check - res(p, i, k)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operation::Elwentwise
{
    void test_interpolate()
    {
        test_interpolate_case1();
        test_interpolate_case2();
        test_interpolate_case3();
    }
}