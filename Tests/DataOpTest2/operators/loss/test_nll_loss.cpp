#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace 
{
    void test_nll_loss_case1()
    {
        cout << "Test NLL loss case 1 (scalar)\t";
        {
            Scalar<CheckElement, CheckDevice> truth(2);
            Scalar<CheckElement, CheckDevice> pred(3);
            auto op = NLLLoss(truth, pred);
            auto res = Evaluate(op);
            
            auto check = -log(3);
            assert(fabs(res.Value() - check) < 0.001f);
        }        
        cout << "done" << endl;
    }

    void test_nll_loss_case2()
    {
        cout << "Test NLL loss case 2 (matrix)\t";
        auto truth = GenTensor<CheckElement>(-100, 3, 10, 7);
        auto pred = GenTensor<CheckElement>(1, 0.1, 10, 7);
        auto op = NLLLoss(truth, pred);
        static_assert(IsScalar<decltype(op)>);
        
        auto res = Evaluate(op);
        static_assert(IsScalar<decltype(res)>);
        
        CheckElement check{};
        CheckElement sumTruth{};
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 7; ++j)
            {
                check -= truth(i, j) * log(pred(i, j));
                sumTruth += truth(i, j);
            }
        }
        assert(fabs(check / sumTruth - res.Value()) < 0.001f);
        cout << "done" << endl;
    }

    void test_nll_loss_case3()
    {
        cout << "Test NLL loss case 2 (batch matrix)\t";
        auto truth = GenTensor<CheckElement>(-100, 3, 3, 10, 7);
        auto pred = GenTensor<CheckElement>(1, 0.1, 3, 10, 7);
        auto op = NLLLoss(truth, pred);
        static_assert(IsScalar<decltype(op)>);

        auto res = Evaluate(op);
        static_assert(IsScalar<decltype(res)>);

        CheckElement check{};
        CheckElement sumTruth{};
        for (size_t b = 0; b < 3; ++b)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 7; ++j)
                {
                    check -= truth(b, i, j) * log(pred(b, i, j));
                    sumTruth += truth(b, i, j);
                }
            }
        }
        assert(fabs(check / sumTruth - res.Value()) < 0.001f);
        cout << "done" << endl;
    }
}

namespace Test::Operators::Loss
{
    void test_nll_loss()
    {
        test_nll_loss_case1();
        test_nll_loss_case2();
        test_nll_loss_case3();
    }
}

namespace
{
    void test_nll_loss_grad1()
    {
        cout << "Test NLL loss grad case 1 ...\t";
        auto rm1 = GenTensor<CheckElement>(1, 1, 4, 5);
        auto rm2 = GenTensor<CheckElement>(2, 2, 4, 5);
        auto div = NLLLossGrad(Scalar<CheckElement, CheckDevice>(0.5), rm1, rm2);
        auto div_r = Evaluate(div);

        CheckElement checkSum{};
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j<5; ++j)
            {
                checkSum += rm1(i, j);
            }
        }
        
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j<5; ++j)
            {
                assert(fabs(div_r(i, j) + 0.5 * rm1(i, j) / rm2(i, j) / checkSum) < 0.001);
            }
        }

        cout << "done" << endl;
    }

    void test_nll_loss_grad2()
    {
        cout << "Test NLL loss grad case 2 ...\t";
        {
            auto rm1 = GenTensor<CheckElement>(1.0f, 0.0001f, 4, 5);
            auto rm2 = GenTensor<CheckElement>(2, 2, 4, 5);
            auto res = NLLLossGrad(Scalar<CheckElement, CheckDevice>(1), rm1, rm2);
            auto res2 = NLLLossGrad(Scalar<CheckElement, CheckDevice>(1), rm1, rm2);

            assert(res == res2);

            auto cm1 = Evaluate(res);
            auto cm2 = Evaluate(res);
            assert(cm1 == cm2);
        }
        {
            auto rm1 = GenTensor<CheckElement>(1.0f, 0.0001f, 4, 5);
            auto rm2 = GenTensor<CheckElement>(2, 2, 4, 5);
            auto res = NLLLossGrad(Scalar<CheckElement, CheckDevice>(1), rm1, rm2);
            auto res2 = res;

            assert(res == res2);

            auto handle1 = res.EvalRegister();
            auto handle2 = res2.EvalRegister();
            EvalPlan<DeviceTags::CPU>::Inst().Eval();

            auto cm1 = handle1.Data();
            auto cm2 = handle2.Data();
            assert(cm1 == cm2);
        }
        cout << "done" << endl;
    }

    void test_nll_loss_grad3()
    {
        cout << "Test NLL loss grad case 3 ...\t";
        auto rm1 = GenTensor<CheckElement>(1, 1, 4, 5, 7);
        auto rm2 = GenTensor<CheckElement>(2, 2, 4, 5, 7);

        auto div = NLLLossGrad(Scalar<CheckElement, CheckDevice>(0.5), rm1, rm2);
        auto div_r = Evaluate(div);

        CheckElement checkSum{};
        for (size_t b = 0; b < 4; ++b)
        {
            for (size_t i = 0; i < 5; ++i)
            {
                for (size_t j = 0; j < 7; ++j)
                {
                    checkSum += rm1(b, i, j);
                }
            }
        }
        
        for (size_t b = 0; b < 4; ++b)
        {
            for (size_t i = 0; i < 5; ++i)
            {
                for (size_t j = 0; j < 7; ++j)
                {
                    assert(fabs(div_r(b, i, j) + 0.5 * rm1(b, i, j) / rm2(b, i, j) / checkSum) < 0.001);
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators::Loss
{
    void test_nll_loss_grad()
    {
        test_nll_loss_grad1();
        test_nll_loss_grad2();
        test_nll_loss_grad3();
    }
}