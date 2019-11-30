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
        auto ori1 = GenMatrix<CheckElement>(10, 7, -100, 3);
        auto ori2 = GenMatrix<CheckElement>(10, 7, 1, 1.5);
        auto lambda = GenMatrix<CheckElement>(10, 7, 0.1, -2);
        auto op = Interpolate(ori1, ori2, lambda);
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
                auto check = ori1(i, k) * lambda(i, k) + ori2(i, k) * (1 - lambda(i, k));
                assert(fabs(check - res(i, k)) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators::Elwentwise
{
    void test_interpolate()
    {
        test_interpolate_case1();
        test_interpolate_case2();
    }
}