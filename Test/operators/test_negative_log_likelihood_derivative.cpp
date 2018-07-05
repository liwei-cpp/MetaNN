#include "test_negative_log_likelihood_derivative.h"
#include "../facilities/data_gen.h"

#include <MetaNN/meta_nn.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_negative_log_likelihood_derivative1()
{
    cout << "Test negative log likelihood derivative case 1 ...\t";
    auto rm1 = GenMatrix<float>(4, 5, 1, 1);
    auto rm2 = GenMatrix<float>(4, 5, 2, 2);
    auto div = NegativeLogLikelihoodDerivative(Scalar<float>(0.5), rm1, rm2);
    auto div_r = Evaluate(div);

    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(fabs(div_r(i, j) + 0.5 * rm1(i, j) / rm2(i, j)) < 0.001);
        }
    }

    rm1 = GenMatrix<float>(111, 113, 1, 1);
    rm2 = GenMatrix<float>(111, 113, 2, 3);
    rm1 = rm1.SubMatrix(31, 35, 17, 22);
    rm2 = rm2.SubMatrix(41, 45, 27, 32);
    div = NegativeLogLikelihoodDerivative(Scalar<float>(0.3), rm1, rm2);
    div_r = Evaluate(div);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(fabs(div_r(i, j) + 0.3 * rm1(i, j) / rm2(i, j)) < 0.001);
        }
    }
    cout << "done" << endl;
}

void test_negative_log_likelihood_derivative2()
{
    cout << "Test negative log likelihood derivative case 2 ...\t";
    {
        auto rm1 = GenMatrix<float>(4, 5, 1.0f, 0.0001f);
        auto rm2 = GenMatrix<float>(4, 5, 2, 2);
        auto res = NegativeLogLikelihoodDerivative(Scalar<float>(1), rm1, rm2);
        auto res2 = NegativeLogLikelihoodDerivative(Scalar<float>(1), rm1, rm2);

        assert(res == res2);

        auto cm1 = Evaluate(res);
        auto cm2 = Evaluate(res);
        assert(cm1 == cm2);
    }
    {
        auto rm1 = GenMatrix<float>(4, 5, 1.0f, 0.0001f);
        auto rm2 = GenMatrix<float>(4, 5, 2, 2);
        auto res = NegativeLogLikelihoodDerivative(Scalar<float>(1), rm1, rm2);
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

void test_negative_log_likelihood_derivative3()
{
    cout << "Test negative log likelihood derivative case 3 ...\t";
    auto rm1 = GenBatchMatrix<float>(4, 5, 7, 1, 1);
    auto rm2 = GenBatchMatrix<float>(4, 5, 7, 2, 2);
    auto div = NegativeLogLikelihoodDerivative(MakeDuplicate(7, Scalar<float>(0.5)), rm1, rm2);
    auto div_r = Evaluate(div);

    for (size_t b = 0; b < 7; ++b)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j<5; ++j)
            {
                assert(fabs(div_r[b](i, j) + 0.5 * rm1[b](i, j) / rm2[b](i, j)) < 0.001);
            }
        }
    }
    cout << "done" << endl;
}
}

void test_negative_log_likelihood_derivative()
{
    test_negative_log_likelihood_derivative1();
    test_negative_log_likelihood_derivative2();
    test_negative_log_likelihood_derivative3();
}
