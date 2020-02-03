#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace 
{
    void test_reduce_sum_case1()
    {
        cout << "Test reduce sum case 1\t";
        auto ori = GenTensor<int>(1, 0, 2, 3);
        {
            auto op = ReduceSum<PolicyContainer<DimArrayIs<1>>>(ori);
            static_assert(IsVector<decltype(op)>);
            assert(op.Shape()[0] == 2);

            auto res = Evaluate(op);
            static_assert(IsVector<decltype(res)>);
            assert(res.Shape()[0] == 2);
            assert(res(0) == 3);
            assert(res(1) == 3);
        }
        {
            auto op = ReduceSum<PolicyContainer<DimArrayIs<0>>>(ori);
            static_assert(IsVector<decltype(op)>);
            assert(op.Shape()[0] == 3);

            auto res = Evaluate(op);
            static_assert(IsVector<decltype(res)>);
            assert(res.Shape()[0] == 3);
            assert(res(0) == 2);
            assert(res(1) == 2);
            assert(res(2) == 2);
        }
        cout << "done" << endl;
    }

    void test_reduce_sum_case2()
    {
        cout << "Test reduce sum case 2\t";
        Vector<CheckElement, CheckDevice> ori(3);
        ori.SetValue(0, 0.1133f);
        ori.SetValue(1, -0.9567f);
        ori.SetValue(2, 0.2958f);

        auto op = ReduceSum<PolicyContainer<DimArrayIs<0>>>(ori);
        static_assert(IsScalar<decltype(op)>);

        auto res = Evaluate(op);
        static_assert(IsScalar<decltype(res)>);
        assert(fabs(res.Value() + 0.5475) < 0.001f);

        cout << "done" << endl;
    }
}

namespace Test::Operators::Math
{
    void test_reduce_sum()
    {
        test_reduce_sum_case1();
        test_reduce_sum_case2();
    }
}