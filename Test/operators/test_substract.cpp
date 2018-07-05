#include "test_substract.h"
#include "../facilities/data_gen.h"
#include <MetaNN/meta_nn.h>
#include <cmath>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_substract1()
{
    cout << "Test substract case 1 ...\t";
    auto rm1 = GenMatrix<int>(4, 5, 0, 1);
    auto rm2 = GenMatrix<int>(4, 5, 3, -1);
    auto sub = rm1 - rm2;
    auto sub_r = Evaluate(sub);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(sub_r(i, j) == rm1(i, j) - rm2(i, j));
        }
    }

    rm1 = GenMatrix<int>(111, 113, 0, 1);
    rm2 = GenMatrix<int>(111, 113, 2, 3);
    rm1 = rm1.SubMatrix(31, 35, 17, 22);
    rm2 = rm2.SubMatrix(41, 45, 27, 32);
    sub = rm1 - rm2;
    sub_r = Evaluate(sub);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(sub_r(i, j) == rm1(i, j) - rm2(i, j));
        }
    }
    cout << "done" << endl;
}

void test_substract2()
{
    cout << "Test substract case 2 ...\t";
    auto rm1 = GenMatrix<int>(4, 5, 3, -1);
    auto sub = rm1 - Scalar<int>(2);
    auto sub_r = Evaluate(sub);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(sub_r(i, j) == rm1(i, j) - 2);
        }
    }

    rm1 = GenMatrix<int>(111, 113, 2, 3);
    rm1 = rm1.SubMatrix(31, 35, 17, 22);
    auto sub1 = Scalar<int>(3) - rm1;
    sub_r = Evaluate(sub1);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(sub_r(i, j) == 3 - rm1(i, j));
        }
    }
    cout << "done" << endl;
}

void test_substract3()
{
    cout << "Test substract case 3 ...\t";
    {
        auto rm1 = GenMatrix<float>(4, 5, 1, 2);
        auto rm2 = GenMatrix<float>(4, 5, 3, 2);
        auto res = rm1 - rm2;
        auto res2 = rm1 - rm2;

        assert(res == res2);

        auto cm1 = Evaluate(res);
        auto cm2 = Evaluate(res);
        assert(cm1 == cm2);
    }
    {
        auto rm1 = GenMatrix<float>(4, 5, 1, 2);
        auto rm2 = GenMatrix<float>(4, 5, 3, 2);
        auto res = rm1 - rm2;
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

void test_substract4()
{
    cout << "Test substract case 4 ...\t";
    {
        auto rm1 = GenBatchMatrix<int>(4, 5, 7, 3, -1);
        auto sub = Scalar<int>(2) - rm1;
        auto sub_r = Evaluate(sub);
        for (size_t b = 0 ; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j<5; ++j)
                {
                    assert(sub_r[b](i, j) == 2 - rm1[b](i, j));
                }
            }
        }
    }
    {
        auto rm1 = GenBatchMatrix<int>(4, 5, 7, 3, -1);
        auto sub = rm1 - Scalar<int>(2);
        auto sub_r = Evaluate(sub);
        for (size_t b = 0 ; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j<5; ++j)
                {
                    assert(sub_r[b](i, j) == rm1[b](i, j) - 2);
                }
            }
        }
    }
    cout << "done" << endl;
}

void test_substract5()
{
    cout << "Test substract case 5 ...\t";
    auto rm1 = GenBatchMatrix<int>(4, 5, 7, 3, -1);
    auto rm2 = GenBatchMatrix<int>(4, 5, 7, 13, 3);
    auto sub = rm1 - rm2;
    auto sub_r = Evaluate(sub);
    for (size_t b = 0 ; b < 7; ++b)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j<5; ++j)
            {
                assert(sub_r[b](i, j) == rm1[b](i, j) - rm2[b](i, j));
            }
        }
    }
    cout << "done" << endl;
}

void test_substract6()
{
    cout << "Test substract case 6 ...\t";
    {
        auto rm1 = GenBatchMatrix<int>(4, 5, 7, 3, -1);
        auto rm2 = GenMatrix<int>(4, 5, 13, 3);
        auto sub = rm1 - rm2;
        auto sub_r = Evaluate(sub);
        for (size_t b = 0 ; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j<5; ++j)
                {
                    assert(sub_r[b](i, j) == rm1[b](i, j) - rm2(i, j));
                }
            }
        }
    }
    
    {
        auto rm1 = GenBatchMatrix<int>(4, 5, 7, 3, -1);
        auto rm2 = GenMatrix<int>(4, 5, 13, 3);
        auto sub = rm2 - rm1;
        auto sub_r = Evaluate(sub);
        for (size_t b = 0 ; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j<5; ++j)
                {
                    assert(sub_r[b](i, j) == rm2(i, j) - rm1[b](i, j));
                }
            }
        }
    }
    cout << "done" << endl;
}
}

void test_substract()
{
    test_substract1();
    test_substract2();
    test_substract3();
    test_substract4();
    test_substract5();
    test_substract6();
}
