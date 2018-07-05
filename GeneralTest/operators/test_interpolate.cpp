#include "test_interpolate.h"
#include "../facilities/data_gen.h"
#include <MetaNN/meta_nn.h>
#include <cassert>
#include <cmath>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_interpolate1()
{
    cout << "Test interpolate case 1 ...\t";
    auto rm1 = GenMatrix<float>(4, 5, 0, 0.0001f);
    auto rm2 = GenMatrix<float>(4, 5, 1, 0.0002f);
    auto rm3 = GenMatrix<float>(4, 5, 2, 0.0007f);
    auto res = Interpolate(rm1, rm2, rm3);
    auto res_r = Evaluate(res);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            float h = rm1(i, j) * rm3(i, j) + rm2(i, j) * (1 - rm3(i, j));
            assert(fabs(res_r(i, j) - h) < 0.0001);
        }
    }

    rm1 = GenMatrix<float>(111, 113, 0, 0.0001f);
    rm2 = GenMatrix<float>(111, 113, 1, 0.0002f);
    rm3 = GenMatrix<float>(111, 113, 3, 0.0007f);

    rm1 = rm1.SubMatrix(31, 35, 17, 22);
    rm2 = rm2.SubMatrix(41, 45, 27, 32);
    rm3 = rm3.SubMatrix(51, 55, 37, 42);

    res = Interpolate(rm1, rm2, rm3);
    res_r = Evaluate(res);

    auto rm4 = GenMatrix<float>(111, 113, 5, 0.001f);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            float h = rm1(i, j) * rm3(i, j) + rm2(i, j) * (1 - rm3(i, j));
            assert(fabs(res_r(i, j) - h) < 0.0001);
        }
    }
    cout << "done" << endl;
}

void test_interpolate2()
{
    cout << "Test interpolate case 2 ...\t";
    {
        auto rm1 = GenMatrix<float>(4, 5, 0, 0.0001f);
        auto rm2 = GenMatrix<float>(4, 5, 1, 0.002f);
        auto rm3 = GenMatrix<float>(4, 5, 0, 0.0001f);
        auto res = Interpolate(rm1, rm2, rm3);
        auto res2 = Interpolate(rm1, rm2, rm3);

        assert(res == res2);

        auto handle1 = res.EvalRegister();
        auto handle2 = res.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto cm1 = handle1.Data();
        auto cm2 = handle2.Data();
        assert(cm1 == cm2);
    }
    {
        auto rm1 = GenMatrix<float>(4, 5, 0, 0.0001f);
        auto rm2 = GenMatrix<float>(4, 5, 1, 0.002f);
        auto rm3 = GenMatrix<float>(4, 5, 0, 0.0001f);
        auto res = Interpolate(rm1, rm2, rm3);
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
}

void test_interpolate()
{
    test_interpolate1();
    test_interpolate2();
}
