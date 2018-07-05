#include "test_tanh_derivative.h"
#include "../facilities/data_gen.h"
#include <MetaNN/meta_nn.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_tanh_derivative1()
{
    cout << "Test tanh derivative case 1 ...\t";
    auto rm1 = GenMatrix<float>(4, 5, 0, 0.0001f);
    auto rm2 = GenMatrix<float>(4, 5, 1, 0.0003f);
    auto t = TanhDerivative(rm1, rm2);
    auto t_r = Evaluate(t);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            float aim = rm1(i, j) * (1 - rm2(i, j) * rm2(i, j));
            assert(fabs(t_r(i, j) - aim) < 0.0001);
        }
    }

    rm1 = GenMatrix<float>(111, 113, 1, 0.001f);
    rm2 = GenMatrix<float>(111, 113, 0, 0.003f);
    rm1 = rm1.SubMatrix(31, 35, 17, 22);
    rm2 = rm2.SubMatrix(30, 34, 18, 23);
    t = TanhDerivative(rm1, rm2);
    t_r = Evaluate(t);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            float aim = rm1(i, j) * (1 - rm2(i, j) * rm2(i, j));
            assert(fabs(t_r(i, j) - aim) < 0.0001);
        }
    }
    cout << "done" << endl;
}

void test_tanh_derivative2()
{
    cout << "Test tanh derivative case 2 ...\t";
    {
        auto rm1 = GenMatrix<float>(4, 5, 0, 0.0001f);
        auto rm2 = GenMatrix<float>(4, 5, 1, 0.0003f);
        auto res = TanhDerivative(rm1, rm2);
        auto res2 = TanhDerivative(rm1, rm2);

        assert(res == res2);

        auto cm1 = Evaluate(res);
        auto cm2 = Evaluate(res);
        assert(cm1 == cm2);
    }
    {
        auto rm1 = GenMatrix<float>(4, 5, 0, 0.0001f);
        auto rm2 = GenMatrix<float>(4, 5, 1, 0.0003f);
        auto res = TanhDerivative(rm1, rm2);
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

void test_tanh_derivative3()
{
    cout << "Test tanh derivative case 3 ...\t";
    auto rm1 = GenBatchMatrix<float>(4, 5, 7, 0, 0.0001f);
    auto rm2 = GenBatchMatrix<float>(4, 5, 7, 1, 0.0003f);
    auto t = TanhDerivative(rm1, rm2);
    auto t_r = Evaluate(t);
    for (size_t b = 0; b < 7; ++b)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j<5; ++j)
            {
                float aim = rm1[b](i, j) * (1 - rm2[b](i, j) * rm2[b](i, j));
                assert(fabs(t_r[b](i, j) - aim) < 0.0001);
            }
        }
    }
    cout << "done" << endl;
}
}

void test_tanh_derivative()
{
    test_tanh_derivative1();
    test_tanh_derivative2();
    test_tanh_derivative3();
}

