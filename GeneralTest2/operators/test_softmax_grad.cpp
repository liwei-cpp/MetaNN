#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_softmax_grad_case1()
    {
        cout << "Test softmax grad case 1 ...\t";
        auto input = Vector<CheckElement, CheckDevice>::CreateWithShape(3);
        input.SetValue(0.5484, 0);
        input.SetValue(0.3292, 1);
        input.SetValue(0.1224, 2);
        
        auto grad = Vector<CheckElement, CheckDevice>::CreateWithShape(3);
        grad.SetValue(0.5911, 0);
        grad.SetValue(0.6659, 1);
        grad.SetValue(0.7868, 2);
        
        auto op = SoftmaxGrad(grad, input);
        auto res = Evaluate(op);
        
        assert(fabs(res(0, 0) + 0.0266) < 0.001f);
        assert(fabs(res(0, 1) - 0.0086) < 0.001f);
        assert(fabs(res(0, 2) - 0.0180) < 0.001f);
        cout << "done" << endl;
    }
    
    void test_softmax_grad_case2()
    {
        cout << "Test softmax grad case 2 ...\t";
        auto input = BatchMatrix<CheckElement, CheckDevice>::CreateWithShape(2, 1, 3);
        input.SetValue(0.5484, 0, 0, 0);
        input.SetValue(0.3292, 0, 0, 1);
        input.SetValue(0.1224, 0, 0, 2);
        
        input.SetValue(0.3915, 1, 0, 0);
        input.SetValue(0.0655, 1, 0, 1);
        input.SetValue(0.5430, 1, 0, 2);
        
        auto grad = BatchMatrix<CheckElement, CheckDevice>::CreateWithShape(2, 1, 3);
        grad.SetValue(0.5911, 0, 0, 0);
        grad.SetValue(0.6659, 0, 0, 1);
        grad.SetValue(0.7868, 0, 0, 2);
        
        grad.SetValue(1.1634, 1, 0, 0);
        grad.SetValue(1.7164, 1, 0, 1);
        grad.SetValue(0.2763, 1, 0, 2);
        
        auto op = SoftmaxGrad(grad, input);
        auto res = Evaluate(op);
        
        assert(fabs(res[0](0, 0) + 0.0266) < 0.001f);
        assert(fabs(res[0](0, 1) - 0.0086) < 0.001f);
        assert(fabs(res[0](0, 2) - 0.0180) < 0.001f);
        
        assert(fabs(res[1](0, 0) - 0.1744) < 0.001f);
        assert(fabs(res[1](0, 1) - 0.0654) < 0.001f);
        assert(fabs(res[1](0, 2) + 0.2398) < 0.001f);
        cout << "done" << endl;
    }
}

namespace Test::Operators
{
    void test_softmax_grad()
    {
        test_softmax_grad_case1();
        test_softmax_grad_case2();
    }
}