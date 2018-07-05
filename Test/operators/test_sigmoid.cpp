#include "test_sigmoid.h"
#include "../facilities/data_gen.h"
#include <MetaNN/meta_nn.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_sigmoid1()
{
    cout << "Test sigmoid case 1 ...\t";
    auto rm1 = GenMatrix<float>(4, 5, 0, 0.0001f);
    auto t = Sigmoid(rm1);
    
    auto handle = t.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();
    auto t_r = handle.Data();

    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            float aim = 1/(1+exp(-rm1(i, j)));
            assert(fabs(t_r(i, j) - aim) < 0.0001);
        }
    }

    rm1 = GenMatrix<float>(111, 113, 2, 0.0001f);
    rm1 = rm1.SubMatrix(31, 35, 17, 22);
    t = Sigmoid(rm1);
    t_r = Evaluate(t);

    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            float aim = 1/(1+exp(-rm1(i, j)));
            assert(fabs(t_r(i, j) - aim) < 0.0001);
        }
    }
    cout << "done" << endl;
}

void test_sigmoid2()
{
    cout << "Test sigmoid case 2 ...\t";
    {
        auto rm1 = GenMatrix<float>(4, 5, 0, 0.0001f);
        auto res = Sigmoid(rm1);
        auto res2 = Sigmoid(rm1);

        assert(res == res2);

        auto cm1 = Evaluate(res);
        auto cm2 = Evaluate(res);
        assert(cm1 == cm2);
    }
    {
        auto rm1 = GenMatrix<float>(4, 5, 0, 0.0001f);
        auto res = Sigmoid(rm1);
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

void test_sigmoid3()
{
    cout << "Test sigmoid case 3 ...\t";
    auto rm1 = GenBatchMatrix<float>(4, 5, 7, 0, 0.0001f);
    auto t = Sigmoid(rm1);
    
    auto handle = t.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();
    auto t_r = handle.Data();

    for (size_t b = 0; b < 7; ++b)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j<5; ++j)
            {
                float aim = 1/(1+exp(-rm1[b](i, j)));
                assert(fabs(t_r[b](i, j) - aim) < 0.0001);
            }
        }
    }
    cout << "done" << endl;
}
}

void test_sigmoid()
{
    test_sigmoid1();
    test_sigmoid2();
    test_sigmoid3();
}
