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
            Scalar<CheckElement, CheckDevice> weight(2);
            Scalar<CheckElement, CheckDevice> input(3);
            auto op = NLLLoss(weight, input);
            auto res = Evaluate(op);
            
            auto check = -2 * log(3);
            assert(fabs(res.Value() - check) < 0.001f);
        }        
        cout << "done" << endl;
    }
    
    void test_nll_loss_case2()
    {
        cout << "Test NLL loss case 2 (matrix)\t";
        auto weight = GenMatrix<CheckElement>(10, 7, -100, 3);
        auto input = GenMatrix<CheckElement>(10, 7, 1, 0.1);
        auto op = NLLLoss(weight, input);
        static_assert(IsScalar<decltype(op)>);
        
        auto res = Evaluate(op);
        static_assert(IsScalar<decltype(res)>);
        
        CheckElement check{};
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 7; ++j)
            {
                check -= weight(i, j) * log(input(i, j));
            }
        }
        assert(fabs(check - res.Value()) < 0.001f);
        cout << "done" << endl;
    }
    
    void test_nll_loss_case3()
    {
        cout << "Test NLL loss case 2 (batch matrix)\t";
        auto weight = GenBatchMatrix<CheckElement>(3, 10, 7, -100, 3);
        auto input = GenBatchMatrix<CheckElement>(3, 10, 7, 1, 0.1);
        auto op = NLLLoss(weight, input);
        static_assert(IsScalar<decltype(op)>);
        
        auto res = Evaluate(op);
        static_assert(IsScalar<decltype(res)>);
        
        CheckElement check{};
        for (size_t b = 0; b < 3; ++b)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 7; ++j)
                {
                    check -= weight[b](i, j) * log(input[b](i, j));
                }
            }
        }
        // batch average
        check /= 3;
        assert(fabs(check - res.Value()) < 0.001f);
        cout << "done" << endl;
    }
    
    void test_nll_loss_case4()
    {
        cout << "Test NLL loss case 2 (batch matrix sequence)\t";
        auto weight = GenBatchMatrixSequence<CheckElement>(std::vector{3, 5, 7}, 10, 7, 0, 0.001);
        auto input = GenBatchMatrixSequence<CheckElement>(std::vector{3, 5, 7}, 10, 7, 1, 0.1);
        auto op = NLLLoss(weight, input);
        static_assert(IsScalar<decltype(op)>);
        
        auto res = Evaluate(op);
        static_assert(IsScalar<decltype(res)>);
        
        CheckElement check{};
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 7; ++j)
            {
                for (size_t b = 0; b < 3; ++b)
                {
                    check -= weight[0][b](i, j) * log(input[0][b](i, j));
                }
                for (size_t b = 0; b < 5; ++b)
                {
                    check -= weight[1][b](i, j) * log(input[1][b](i, j));
                }
                for (size_t b = 0; b < 7; ++b)
                {
                    check -= weight[2][b](i, j) * log(input[2][b](i, j));
                }
            }
        }
        // average
        check /= (3 + 5 + 7);
        assert(fabs(check - res.Value()) < 0.001f);
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
        test_nll_loss_case4();
    }
}

namespace
{
    void test_nll_loss_grad1()
    {
        cout << "Test NLL loss grad case 1 ...\t";
        auto rm1 = GenMatrix<CheckElement>(4, 5, 1, 1);
        auto rm2 = GenMatrix<CheckElement>(4, 5, 2, 2);
        auto div = NLLLossGrad(Scalar<CheckElement, CheckDevice>(0.5), rm1, rm2);
        auto div_r = Evaluate(div);

        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j<5; ++j)
            {
                assert(fabs(div_r(i, j) + 0.5 * rm1(i, j) / rm2(i, j)) < 0.001);
            }
        }

        cout << "done" << endl;
    }
    
    void test_nll_loss_grad2()
    {
        cout << "Test NLL loss grad case 2 ...\t";
        {
            auto rm1 = GenMatrix<CheckElement>(4, 5, 1.0f, 0.0001f);
            auto rm2 = GenMatrix<CheckElement>(4, 5, 2, 2);
            auto res = NLLLossGrad(Scalar<CheckElement, CheckDevice>(1), rm1, rm2);
            auto res2 = NLLLossGrad(Scalar<CheckElement, CheckDevice>(1), rm1, rm2);

            assert(res == res2);

            auto cm1 = Evaluate(res);
            auto cm2 = Evaluate(res);
            assert(cm1 == cm2);
        }
        {
            auto rm1 = GenMatrix<CheckElement>(4, 5, 1.0f, 0.0001f);
            auto rm2 = GenMatrix<CheckElement>(4, 5, 2, 2);
            auto res = NLLLossGrad(Scalar<CheckElement, CheckDevice>(1), rm1, rm2);
            auto res2 = res;

            assert(res == res2);

            auto handle1 = res.EvalRegister();
            auto handle2 = res2.EvalRegister();
            EvalPlan<DeviceTags::CPU>::Eval();

            auto cm1 = handle1.Data();
            auto cm2 = handle2.Data();
            assert(cm1 == cm2);
        }
        cout << "done" << endl;
    }
    
    void test_nll_loss_grad3()
    {
        cout << "Test NLL loss grad case 3 ...\t";
        auto rm1 = GenBatchMatrix<CheckElement>(4, 5, 7, 1, 1);
        auto rm2 = GenBatchMatrix<CheckElement>(4, 5, 7, 2, 2);
        
        auto grad = Duplicate(Scalar<CheckElement, CheckDevice>(0.5), Shape<CategoryTags::BatchScalar>(7));
        auto div = NLLLossGrad(grad, rm1, rm2);
        auto div_r = Evaluate(div);

        for (size_t b = 0; b < 4; ++b)
        {
            for (size_t i = 0; i < 5; ++i)
            {
                for (size_t j = 0; j < 7; ++j)
                {
                    assert(fabs(div_r[b](i, j) + 0.5 * rm1[b](i, j) / rm2[b](i, j)) < 0.001);
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