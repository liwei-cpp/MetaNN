#include "test_softmax.h"

#include "../facilities/data_gen.h"
#include <MetaNN/meta_nn.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_softmax1()
{
    cout << "Test softmax case 1 ...\t";
    auto rm1 = GenMatrix<float>(1, 20, 0, 0.001f);
    auto t = VecSoftmax(rm1);
    auto t_r = Evaluate(t);

    float sum = 0;
    for (size_t i = 0; i < 20; ++i)
    {
        sum += exp(rm1(0, i));
    }

    for (size_t i = 0; i < 20; ++i)
    {
        assert(fabs(t_r(0, i) - exp(rm1(0, i)) / sum) < 0.0001);
    }

    rm1 = GenMatrix<float>(111, 113, 2, 0.001f);
    rm1 = rm1.SubMatrix(17, 18, 31, 51);
    t = VecSoftmax(rm1);
    t_r = Evaluate(t);

    sum = 0;
    for (size_t i = 0; i < 20; ++i)
    {
        sum += exp(rm1(0, i));
    }

    for (size_t i = 0; i < 20; ++i)
    {
        assert(fabs(t_r(0, i) - exp(rm1(0, i)) / sum) < 0.0001);
    }
    cout << "done" << endl;
}

void test_softmax2()
{
    cout << "Test softmax case 2 ...\t";
    {
        auto rm1 = GenMatrix<float>(1, 20, 0, 0.001f);
        auto res = VecSoftmax(rm1);
        auto res2 = VecSoftmax(rm1);

        assert(res == res2);

        auto cm1 = Evaluate(res);
        auto cm2 = Evaluate(res);
        assert(cm1 == cm2);
    }
    {
        auto rm1 = GenMatrix<float>(1, 20, 0, 0.001f);
        auto res = VecSoftmax(rm1);
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

void test_softmax3()
{
    cout << "Test softmax case 3 ...\t";
    auto rm1 = GenBatchMatrix<float>(1, 20, 7, 0, 0.001f);
    auto t = VecSoftmax(rm1);
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

void test_softmax()
{
    test_softmax1();
    test_softmax2();
    test_softmax3();
}
