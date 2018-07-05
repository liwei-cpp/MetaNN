#include "test_dot.h"
#include "../facilities/data_gen.h"
#include <MetaNN/meta_nn.h>
#include <iostream>
#include <cassert>
using namespace std;
using namespace MetaNN;

namespace
{
void test_dot_1()
{
    cout << "Test dot case 1 ...\t";
    auto rm = GenMatrix<int>(4, 5, 0, 1);
    auto cm = GenMatrix<int>(5, 3, 3, 2);
    auto mul = Dot(rm, cm);
    auto mul_r = Evaluate(mul);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<3; ++j)
        {
            int h = 0;
            for (size_t k = 0; k < 5; ++k)
            {
                h += rm(i,k) * cm(k, j);
            }
            assert(h == mul_r(i, j));
        }
    }

    auto rm2 = GenMatrix<int>(111, 113, 0, 1);
    auto cm2 = GenMatrix<int>(111, 113, 2, 3);
    rm2 = rm2.SubMatrix(31, 35, 17, 22);
    cm2 = cm2.SubMatrix(31, 36, 41, 44);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            rm2.SetValue(i, j, rm(i, j));
        }
    }
    for (size_t i = 0; i < 5; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            cm2.SetValue(i, j, cm(i, j));
        }
    }
    auto mul2 = Dot(rm2, cm2);
    auto mul2_r = Evaluate(mul2);

    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<3; ++j)
        {
            assert(mul2_r(i, j) == mul_r(i, j));
        }
    }
    cout << "done" << endl;
}

void test_dot_2()
{
    cout << "Test dot case 2 ...\t";
    auto rm = GenBatchMatrix<int>(4, 5, 3, 0, 1);
    auto cm = GenMatrix<int>(5, 3, 3, 2);
    auto mul = Dot(rm, cm);
    auto mul_r = Evaluate(mul);
    for (size_t b = 0; b < 3; ++b)
    {
        auto rm1 = rm[b];
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j<3; ++j)
            {
                int h = 0;
                for (size_t k = 0; k < 5; ++k)
                {
                    h += rm1(i,k) * cm(k, j);
                }
                assert(h == mul_r[b](i, j));
            }
        }
    }
    cout << "done" << endl;
}

void test_dot_3()
{
    cout << "Test dot case 3 ...\t";
    auto rm = GenMatrix<int>(4, 5, 0, 1);
    auto cm = GenBatchMatrix<int>(5, 3, 3, 3, 2);
    auto mul = Dot(rm, cm);
    auto mul_r = Evaluate(mul);
    for (size_t b = 0; b < 3; ++b)
    {
        auto cm1 = cm[b];
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j<3; ++j)
            {
                int h = 0;
                for (size_t k = 0; k < 5; ++k)
                {
                    h += rm(i,k) * cm1(k, j);
                }
                assert(h == mul_r[b](i, j));
            }
        }
    }
    cout << "done" << endl;
}
}

void test_dot()
{
    test_dot_1();
    test_dot_2();   // BatchMatrix dot matrix
    test_dot_3();
}
