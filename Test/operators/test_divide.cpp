#include "test_divide.h"
#include "../facilities/data_gen.h"
#include <MetaNN/meta_nn.h>
#include <cmath>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_div1()
{
    cout << "Test divide case 1 ...\t";
    auto rm1 = GenMatrix<float>(4, 5, 1, 1);
    auto rm2 = GenMatrix<float>(4, 5, 2, 2);
    auto div = rm1 / rm2;
    auto div_r = Evaluate(div);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(fabs(div_r(i, j) - rm1(i, j) / rm2(i, j)) < 0.001);
        }
    }

    rm1 = GenMatrix<float>(111, 113, 1, 1);
    rm2 = GenMatrix<float>(111, 113, 2, 3);
    rm1 = rm1.SubMatrix(31, 35, 17, 22);
    rm2 = rm2.SubMatrix(41, 45, 27, 32);
    div = rm1 / rm2;

    div_r = Evaluate(div);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(fabs(div_r(i, j) - rm1(i, j) / rm2(i, j)) < 0.001);
        }
    }
    cout << "done" << endl;
}

void test_div2()
{
    cout << "Test divide case 2 ...\t";
    auto rm1 = GenMatrix<float>(4, 5, 1, 1);
    auto div = rm1 / Scalar<float>(2);
    auto div_r = Evaluate(div);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(fabs(div_r(i, j) - rm1(i, j) / 2) < 0.001);
        }
    }

    rm1 = GenMatrix<float>(111, 113, 2, 3);
    rm1 = rm1.SubMatrix(31, 35, 17, 22);
    auto div1 = Scalar<float>(3) / rm1;

    div_r = Evaluate(div1);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(fabs(div_r(i, j) - 3 / rm1(i, j)) < 0.001);
        }
    }
    cout << "done" << endl;
}

void test_div3()
{
    cout << "Test divide case 3 ...\t";
    auto rm1 = GenBatchMatrix<float>(4, 5, 7, 1, 1);
    auto rm2 = GenBatchMatrix<float>(4, 5, 7, 2, 2);
    auto div = rm1 / rm2;
    auto div_r = Evaluate(div);
    for (size_t b = 0; b < 7; ++b)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j<5; ++j)
            {
                assert(fabs(div_r[b](i, j) - rm1[b](i, j) / rm2[b](i, j)) < 0.001);
            }
        }
    }

    cout << "done" << endl;
}

void test_div4()
{
    cout << "Test divide case 4 ...\t";
    {
        auto rm1 = GenBatchMatrix<float>(4, 5, 7, 1, 1);
        auto rm2 = GenMatrix<float>(4, 5, 2, 2);
        auto div = rm1 / rm2;
        auto div_r = Evaluate(div);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j<5; ++j)
                {
                    assert(fabs(div_r[b](i, j) - rm1[b](i, j) / rm2(i, j)) < 0.001);
                }
            }
        }
    }
    
    {
        auto rm1 = GenBatchMatrix<float>(4, 5, 7, 1, 1);
        auto rm2 = GenMatrix<float>(4, 5, 2, 2);
        auto div = rm2 / rm1;
        auto div_r = Evaluate(div);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j<5; ++j)
                {
                    assert(fabs(div_r[b](i, j) - rm2(i, j) / rm1[b](i, j)) < 0.001);
                }
            }
        }
    }

    cout << "done" << endl;
}

void test_div5()
{
    cout << "Test divide case 5 ...\t";
    {
        auto rm1 = GenBatchMatrix<float>(4, 5, 7, 1, 1);
        auto rm2 = Scalar<int>(3);
        auto div = rm1 / rm2;
        auto div_r = Evaluate(div);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j<5; ++j)
                {
                    assert(fabs(div_r[b](i, j) - rm1[b](i, j) / 3) < 0.001);
                }
            }
        }
    }
    
    {
        auto rm1 = GenBatchMatrix<float>(4, 5, 7, 1, 1);
        auto rm2 = Scalar<int>(3);
        auto div = rm2 / rm1;
        auto div_r = Evaluate(div);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j<5; ++j)
                {
                    assert(fabs(div_r[b](i, j) - 3 / rm1[b](i, j)) < 0.001);
                }
            }
        }
    }

    cout << "done" << endl;
}
}

void test_divide()
{
    test_div1();
    test_div2();
    test_div3();
    test_div4();
    test_div5();
}
