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
        cout << "Test reduce sum case 1 (Matrix)\t";
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
        cout << "Test reduce sum case 2 (Vector)\t";
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
    
    void test_reduce_sum_case3()
    {
        cout << "Test reduce sum case 3 (Matrix, Keep Dim)\t";
        auto ori = GenTensor<int>(1, 0, 2, 3);
        {
            auto op = ReduceSum<PolicyContainer<DimArrayIs<1>, PKeepDim>>(ori);
            static_assert(IsMatrix<decltype(op)>);
            assert(op.Shape()[0] == 2);
            assert(op.Shape()[1] == 1);

            auto res = Evaluate(op);
            static_assert(IsMatrix<decltype(res)>);
            assert(res.Shape()[0] == 2);
            assert(res.Shape()[1] == 1);
            assert(res(0, 0) == 3);
            assert(res(1, 0) == 3);
        }
        {
            auto op = ReduceSum<PolicyContainer<DimArrayIs<0>, PKeepDim>>(ori);
            static_assert(IsMatrix<decltype(op)>);
            assert(op.Shape()[0] == 1);
            assert(op.Shape()[1] == 3);

            auto res = Evaluate(op);
            static_assert(IsMatrix<decltype(res)>);
            assert(res.Shape()[0] == 1);
            assert(res.Shape()[1] == 3);
            assert(res(0, 0) == 2);
            assert(res(0, 1) == 2);
            assert(res(0, 2) == 2);
        }
        cout << "done" << endl;
    }
    
    void test_reduce_sum_case4()
    {
        cout << "Test reduce sum case 4 (Vector, Keep Dim)\t";
        Vector<CheckElement, CheckDevice> ori(3);
        ori.SetValue(0, 0.1133f);
        ori.SetValue(1, -0.9567f);
        ori.SetValue(2, 0.2958f);

        auto op = ReduceSum<PolicyContainer<DimArrayIs<0>, PKeepDim>>(ori);
        static_assert(IsVector<decltype(op)>);

        auto res = Evaluate(op);
        static_assert(IsVector<decltype(res)>);
        assert(fabs(res(0) + 0.5475) < 0.001f);

        cout << "done" << endl;
    }
}

namespace Test::Operators::Math
{
    void test_reduce_sum()
    {
        test_reduce_sum_case1();
        test_reduce_sum_case2();
        test_reduce_sum_case3();
        test_reduce_sum_case4();
    }
}