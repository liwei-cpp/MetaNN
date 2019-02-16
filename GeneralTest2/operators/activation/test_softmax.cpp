#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_softmax_case1()
    {
        cout << "Test softmax case 1 ...\t";
        auto input = GenMatrix<float>(1, 20, 0, 0.001f);
        auto op = Softmax(input);
        auto res = Evaluate(op);

        float sum = 0;
        for (size_t i = 0; i < 20; ++i)
        {
            sum += exp(input(0, i));
        }

        for (size_t i = 0; i < 20; ++i)
        {
            assert(fabs(res(0, i) - exp(input(0, i)) / sum) < 0.0001);
        }

        cout << "done" << endl;
    }
    
    void test_softmax_case2()
    {
        cout << "Test softmax case 2 ...\t";
        {
            auto rm1 = GenMatrix<float>(1, 20, 0, 0.001f);
            auto res = Softmax(rm1);
            auto res2 = Softmax(rm1);

            assert(res == res2);

            auto cm1 = Evaluate(res);
            auto cm2 = Evaluate(res);
            assert(cm1 == cm2);
        }
        {
            auto rm1 = GenMatrix<float>(1, 20, 0, 0.001f);
            auto res = Softmax(rm1);
            auto res2 = res;

            assert(res == res2);

            const auto& evalHandle1 = res.EvalRegister();
            const auto& evalHandle2 = res2.EvalRegister();
            EvalPlan<DeviceTags::CPU>::Eval();

            auto cm1 = evalHandle1.Data();
            auto cm2 = evalHandle2.Data();
            assert(cm1 == cm2);
        }
        cout << "done" << endl;
    }
    
    void test_softmax_case3()
    {
        cout << "Test softmax case 3 ...\t";
        auto rm1 = GenBatchMatrix<float>(7, 1, 20, 0, 0.001f);
        auto t = Softmax(rm1);
        auto t_r = Evaluate(t);

        for (size_t b = 0; b < 7; ++b)
        {
            float sum = 0;
            for (size_t i = 0; i < 20; ++i)
            {
                sum += exp(rm1[b](0, i));
            }

            for (size_t i = 0; i < 20; ++i)
            {
                assert(fabs(t_r[b](0, i) - exp(rm1[b](0, i)) / sum) < 0.0001);
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators
{
    void test_softmax()
    {
        test_softmax_case1();
        test_softmax_case2();
        test_softmax_case3();
    }
}

namespace
{
    void test_softmax_grad_case1()
    {
        cout << "Test softmax grad case 1 ...\t";
        Vector<CheckElement, CheckDevice> input(3);
        input.SetValue(0, 0.5484);
        input.SetValue(1, 0.3292);
        input.SetValue(2, 0.1224);
        
        Vector<CheckElement, CheckDevice> grad(3);
        grad.SetValue(0, 0.5911);
        grad.SetValue(1, 0.6659);
        grad.SetValue(2, 0.7868);
        
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
        BatchMatrix<CheckElement, CheckDevice> input(2, 1, 3);
        input.SetValue(0, 0, 0, 0.5484);
        input.SetValue(0, 0, 1, 0.3292);
        input.SetValue(0, 0, 2, 0.1224);
        
        input.SetValue(1, 0, 0, 0.3915);
        input.SetValue(1, 0, 1, 0.0655);
        input.SetValue(1, 0, 2, 0.5430);
        
        BatchMatrix<CheckElement, CheckDevice> grad(2, 1, 3);
        grad.SetValue(0, 0, 0, 0.5911);
        grad.SetValue(0, 0, 1, 0.6659);
        grad.SetValue(0, 0, 2, 0.7868);
        
        grad.SetValue(1, 0, 0, 1.1634);
        grad.SetValue(1, 0, 1, 1.7164);
        grad.SetValue(1, 0, 2, 0.2763);
        
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
    
    void test_softmax_grad_case3()
    {
        cout << "Test softmax grad case 3 (NLL with one hot weight vector) ...\t";

        auto softmaxRes = GenMatrix<CheckElement>(1, 20, 0, 0.001f);
        auto grad = Scalar<CheckElement, CheckDevice>(0.7f);
        auto weight = OneHotVector<CheckElement, CheckDevice>(20, 3);
        auto nllLossBP = NLLLossGrad(grad, weight, softmaxRes);
        auto softmaxBP = SoftmaxGrad(nllLossBP, softmaxRes);
        auto check = Evaluate(softmaxBP);
        for (size_t i = 0; i < 20; ++i)
        {
            CheckElement compare = softmaxRes(0, i);
            if (i == 3)
            {
                compare -= 1;
            }
            assert(fabs(check(0, i) - compare * 0.7f) <= 0.0001);
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators
{
    void test_softmax_grad()
    {
        test_softmax_grad_case1();
        test_softmax_grad_case2();
        test_softmax_grad_case3();
    }
}