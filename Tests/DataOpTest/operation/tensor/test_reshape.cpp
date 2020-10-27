#include <data_gen.h>
#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_reshape_case1()
    {
        cout << "Test reshape case 1 (scalar) ...\t";
        auto ori = GenTensor<CheckElement>(0, 1, 1);
        
        auto op1 = Reshape(ori, Shape());
        static_assert(IsScalar<decltype(op1)>);
        auto res1 = Evaluate(op1);
        static_assert(IsScalar<decltype(res1)>);
        assert(res1.Value() == 0);
        
        auto op2 = Reshape(op1, Shape(-1));
        static_assert(IsVector<decltype(op2)>);
        auto res2 = Evaluate(op2);
        static_assert(IsVector<decltype(res2)>);
        assert(res2(0) == 0);
        
        auto op3 = Reshape(op1, Shape(-1, 1));
        static_assert(IsMatrix<decltype(op3)>);
        auto res3 = Evaluate(op3);
        static_assert(IsMatrix<decltype(res3)>);
        assert(res3(0, 0) == 0);
        cout << "done" << endl;
    }
    
    void test_reshape_case2()
    {
        cout << "Test reshape case 2 ...\t";
        Tensor<int, DeviceTags::CPU, 1> ori(4);
        ori.SetValue(0, 0);
        ori.SetValue(1, 1);
        ori.SetValue(2, 2);
        ori.SetValue(3, 3);
        
        auto op = Reshape(ori, Shape(2, 2));
        static_assert(IsMatrix<decltype(op)>);
        auto res = Evaluate(op);
        static_assert(IsMatrix<decltype(res)>);
        assert(res(0, 0) == 0);
        assert(res(0, 1) == 1);
        assert(res(1, 0) == 2);
        assert(res(1, 1) == 3);
        
        auto op2 = Reshape(op, Shape(-1));
        static_assert(IsVector<decltype(op2)>);
        auto res2 = Evaluate(op2);
        assert(res2(0) == 0);
        assert(res2(1) == 1);
        assert(res2(2) == 2);
        assert(res2(3) == 3);
        
        cout << "done" << endl;
    }
}

namespace Test::Operation::Tensor
{
    void test_reshape()
    {
        test_reshape_case1();
        test_reshape_case2();
    }
}