#include "test_transpose.h"
#include "../facilities/data_gen.h"
#include <MetaNN/meta_nn.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_transpose1()
{
    cout << "Test transpose case 1 ...\t";
    auto rm1 = GenMatrix<int>(4, 5, 0, 1);
    auto tr = Transpose(rm1);
    auto tr_r = Evaluate(tr);
    for (size_t i = 0; i < 5; ++i)
    {
        for (size_t j = 0; j<4; ++j)
        {
            assert(tr_r(i, j) == rm1(j, i));
        }
    }

    rm1 = GenMatrix<int>(111, 113, 2, 3);
    rm1 = rm1.SubMatrix(31, 35, 17, 22);
    tr = Transpose(rm1);
    tr_r = Evaluate(tr);
    for (size_t i = 0; i < 5; ++i)
    {
        for (size_t j = 0; j<4; ++j)
        {
            assert(tr_r(i, j) == rm1(j, i));
        }
    }
    cout << "done" << endl;
}

void test_transpose2()
{
    cout << "Test transpose case 2 ...\t";
    {
        auto rm1 = GenMatrix<int>(4, 5, 0, 1);
        auto res = Transpose(rm1);
        auto res2 = Transpose(rm1);

        assert(res == res2);

        auto cm1 = Evaluate(res);
        auto cm2 = Evaluate(res);
        assert(cm1 == cm2);
    }
    {
        auto rm1 = GenMatrix<int>(4, 5, 0, 1);
        auto res = Transpose(rm1);
        auto res2 = res;

        assert(res == res2);

        const auto& evalHandle1 = res.EvalRegister();
        const auto& evalHandle2 = res2.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto cm1 = evalHandle1.Data();
        auto cm2 = evalHandle2.Data();
    }
    cout << "done" << endl;
}

void test_transpose3()
{
    cout << "Test transpose case 3 ...\t";
    auto rm1 = GenBatchMatrix<int>(4, 5, 7, 0, 1);
    auto tr = Transpose(rm1);
    auto tr_r = Evaluate(tr);
    for (size_t b = 0; b < 7; ++b)
    {
        for (size_t i = 0; i < 5; ++i)
        {
            for (size_t j = 0; j<4; ++j)
            {
                assert(tr_r[b](i, j) == rm1[b](j, i));
            }
        }
    }
    cout << "done" << endl;
}
}

void test_transpose()
{
    test_transpose1();
    test_transpose2();
    test_transpose3();
}
