#include "test_negative_log_likelihood.h"
#include "../facilities/data_gen.h"
#include <MetaNN/meta_nn.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_negative_log_likelihood1()
{
    cout << "Test negative log likelihood case 1 ...\t";
    auto rm1 = GenMatrix<float>(4, 5, 1.0f, 0.0001f);
    auto rm2 = GenMatrix<float>(4, 5, 0.1f, 0.2f);
    auto t = NegativeLogLikelihood(rm1, rm2);
    auto t_r = Evaluate(t);

    float check = 0;
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            check -= rm1(i, j) * log(rm2(i, j));
        }
    }
    assert(fabs(t_r.Value() - check) < 0.0001);


    rm1 = GenMatrix<float>(111, 113, 1.1f, 0.0001f);
    rm1 = rm1.SubMatrix(31, 35, 17, 22);
    t = NegativeLogLikelihood(rm1, rm2);
    t_r = Evaluate(t);

    check = 0;
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            check -= rm1(i, j) * log(rm2(i, j));
        }
    }
    assert(fabs(t_r.Value() - check) < 0.0001);

    cout << "done" << endl;
}

void test_negative_log_likelihood2()
{
    cout << "Test negative log likelihood case 2 ...\t";
    {
        auto rm1 = GenMatrix<float>(4, 5, 1.0f, 0.0001f);
        auto rm2 = GenMatrix<float>(4, 5, 0.1f, 0.2f);
        auto res = NegativeLogLikelihood(rm1, rm2);
        auto res2 = NegativeLogLikelihood(rm1, rm2);

        assert(res == res2);

        auto cm1 = Evaluate(res);
        auto cm2 = Evaluate(res);
        assert(cm1 == cm2);
    }
    {
        auto rm1 = GenMatrix<float>(4, 5, 1.0f, 0.0001f);
        auto rm2 = GenMatrix<float>(4, 5, 0.1f, 0.2f);
        auto res = NegativeLogLikelihood(rm1, rm2);
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

void test_negative_log_likelihood3()
{
    cout << "Test negative log likelihood case 3 ...\t";
    auto rm1 = GenBatchMatrix<float>(4, 5, 7, 1.0f, 0.0001f);
    auto rm2 = GenBatchMatrix<float>(4, 5, 7, 0.1f, 0.2f);
    auto t = NegativeLogLikelihood(rm1, rm2);
    auto t_r = Evaluate(t);

    for (size_t b = 0; b < 7; ++b)
    {
        float check = 0;
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j<5; ++j)
            {
                check -= rm1[b](i, j) * log(rm2[b](i, j));
            }
        }
        assert(fabs(t_r[b] - check) < 0.0001);
    }
    cout << "done" << endl;
}
}

void test_negative_log_likelihood()
{
    test_negative_log_likelihood1();
    test_negative_log_likelihood2();
    test_negative_log_likelihood3();
}
