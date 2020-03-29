#include <data_gen.h>
#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_dot_case1()
    {
        cout << "Test dot case 1 (matrix)\t";
        auto in1 = GenTensor<CheckElement>(0, 1, 5, 3);
        auto in2 = GenTensor<CheckElement>(-10, 0.1, 3, 8);
        auto op = Dot(in1, in2);
        static_assert(IsMatrix<decltype(op)>);
        assert(op.Shape()[0] == 5);
        assert(op.Shape()[1] == 8);
        
        auto res = Evaluate(op);
        static_assert(IsMatrix<decltype(res)>);
        assert(res.Shape()[0] == 5);
        assert(res.Shape()[1] == 8);
        
        for (size_t i = 0; i < 5; ++i)
        {
            for (size_t j = 0; j < 8; ++j)
            {
                CheckElement value = 0;
                for (size_t k = 0; k < 3; ++k)
                {
                    value += in1(i, k) * in2(k, j);
                }
                assert(fabs(value - res(i, j)) < 0.001f);
            }
        }
        cout << "done" << endl;
    }

    void test_dot_case2()
    {
        cout << "Test dot case 2 (vector)\t";
        auto in1 = GenTensor<CheckElement>(0, 1, 5);
        auto in2 = GenTensor<CheckElement>(-10, 0.1, 5);
        auto op = Dot(in1, in2);
        static_assert(IsScalar<decltype(op)>);

        auto res = Evaluate(op);
        static_assert(IsScalar<decltype(res)>);

        CheckElement value = 0;
        for (size_t i = 0; i < 5; ++i)
        {
            value += in1(i) * in2(i);
        }
        assert(fabs(value - res.Value()) < 0.001f);
        cout << "done" << endl;
    }
    
    void test_dot_case3()
    {
        cout << "Test dot case 3 (2 dim dot)\t";
        auto in1 = GenTensor<size_t>(0, 1, 5, 4, 3);
        auto in2 = GenTensor<size_t>(0, 1, 4, 3, 2);
        auto op = Dot<PolicyContainer<PModifyDimNumIs<2>>>(in1, in2);
        static_assert(IsMatrix<decltype(op)>);
        assert(op.Shape()[0] == 5);
        assert(op.Shape()[1] == 2);

        auto res = Evaluate(op);
        static_assert(IsMatrix<decltype(res)>);
        assert(res.Shape()[0] == 5);
        assert(res.Shape()[1] == 2);

        assert(res(0, 0) == 1012);
        assert(res(0, 1) == 1078);
        
        assert(res(1, 0) == 2596);
        assert(res(1, 1) == 2806);
        
        assert(res(2, 0) == 4180);
        assert(res(2, 1) == 4534);
        
        assert(res(3, 0) == 5764);
        assert(res(3, 1) == 6262);
        
        assert(res(4, 0) == 7348);
        assert(res(4, 1) == 7990);
        cout << "done" << endl;
    }
    
    void test_dot_case4()
    {
        cout << "Test dot case 4 (2 dim dot with diff input dim)\t";
        auto in1 = GenTensor<size_t>(0, 1, 5, 4, 3);
        auto in2 = GenTensor<size_t>(0, 1, 4, 3);
        auto op = Dot<PolicyContainer<PModifyDimNumIs<2>>>(in1, in2);
        static_assert(IsVector<decltype(op)>);
        assert(op.Shape()[0] == 5);

        auto res = Evaluate(op);
        static_assert(IsVector<decltype(res)>);
        assert(res.Shape()[0] == 5);

        assert(res(0) == 506);
        assert(res(1) == 1298);
        assert(res(2) == 2090);
        assert(res(3) == 2882);
        assert(res(4) == 3674);
        cout << "done" << endl;
    }
}

namespace Test::Operation::Blas
{
    void test_dot()
    {
        test_dot_case1();
        test_dot_case2();
        test_dot_case3();
        test_dot_case4();
    }
}