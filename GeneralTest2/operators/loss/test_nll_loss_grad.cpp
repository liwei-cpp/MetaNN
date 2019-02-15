#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

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

namespace Test::Operators
{
    void test_nll_loss_grad()
    {
        test_nll_loss_grad1();
        test_nll_loss_grad2();
        test_nll_loss_grad3();
    }
}