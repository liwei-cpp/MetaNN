#include "test_element_mul.h"
#include "../facilities/data_gen.h"
#include <MetaNN/meta_nn.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_mul1()
{
    cout << "Test element mul case 1 ...\t";
    auto rm1 = GenMatrix<int>(4, 5, 0, 1);
    auto rm2 = GenMatrix<int>(4, 5, 3, 2);
    auto mul = rm1 * rm2;
    auto mul_r = Evaluate(mul);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(mul_r(i, j) == rm1(i, j) * rm2(i, j));
        }
    }

    rm1 = GenMatrix<int>(111, 113, 4, 2);
    rm2 = GenMatrix<int>(111, 113, 1, 1);
    rm1 = rm1.SubMatrix(31, 35, 17, 22);
    rm2 = rm2.SubMatrix(41, 45, 27, 32);
    mul = rm1 * rm2;
    mul_r = Evaluate(mul);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(mul_r(i, j) == rm1(i, j) * rm2(i, j));
        }
    }
    cout << "done" << endl;
}

void test_mul2()
{
    cout << "Test element mul case 2 ...\t";
    auto rm1 = GenMatrix<int>(4, 5, 0, 1);
    auto mul = rm1 * Scalar<int>(2);
    auto mul_r = Evaluate(mul);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(mul_r(i, j) == rm1(i, j) * 2);
        }
    }

    rm1 = GenMatrix<int>(111, 113, 2, 3);
    rm1 = rm1.SubMatrix(31, 35, 17, 22);
    mul = Scalar<int>(3) * rm1;

    mul_r = Evaluate(mul);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(mul_r(i, j) == rm1(i, j) * 3);
        }
    }
    cout << "done" << endl;
}

void test_mul3()
{
    cout << "Test element mul case 3 ...\t";
    {
        auto rm1 = GenMatrix<int>(4, 5, 0, 1);
        auto rm2 = GenMatrix<int>(4, 5, 1, 3);
        auto mul = rm1 * rm2;
        auto mul2 = rm1 * rm2;

        assert(mul == mul2);

        auto handle1 = mul.EvalRegister();
        auto handle2 = mul.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto cm1 = handle1.Data();
        auto cm2 = handle2.Data();
        assert(cm1 == cm2);
    }
    {
        auto rm1 = GenMatrix<int>(4, 5, 0, 1);
        auto rm2 = GenMatrix<int>(4, 5, 1, 3);
        auto mul = rm1 * rm2;
        auto mul2 = mul;

        assert(mul == mul2);

        auto handle1 = mul.EvalRegister();
        auto handle2 = mul2.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto cm1 = handle1.Data();
        auto cm2 = handle2.Data();
        assert(cm1 == cm2);
    }
    cout << "done" << endl;
}

void test_mul4()
{
    cout << "Test element mul case 4 ...\t";
    {
        auto rm1 = GenMatrix<int>(4, 5, 0, 1);
        auto rm2 = GenBatchMatrix<int>(4, 5, 7, 3, 2);
        auto mul = rm1 * rm2;
        auto mul_r = Evaluate(mul);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j<5; ++j)
                {
                    assert(mul_r[b](i, j) == rm1(i, j) * rm2[b](i, j));
                }
            }
        }
    }
    
    {
        auto rm1 = GenMatrix<int>(4, 5, 0, 1);
        auto rm2 = GenBatchMatrix<int>(4, 5, 7, 3, 2);
        auto mul = rm2 * rm1;
        auto mul_r = Evaluate(mul);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j<5; ++j)
                {
                    assert(mul_r[b](i, j) == rm1(i, j) * rm2[b](i, j));
                }
            }
        }
    }
    cout << "done" << endl;
}

void test_mul5()
{
    cout << "Test element mul case 5 ...\t";
    {
        auto rm1 = GenBatchMatrix<int>(4, 5, 7, 0, 1);
        auto rm2 = Scalar<int>(13);
        auto mul = rm1 * rm2;
        auto mul_r = Evaluate(mul);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j<5; ++j)
                {
                    assert(mul_r[b](i, j) == rm1[b](i, j) * 13);
                }
            }
        }
    }
    
    {
        auto rm1 = GenBatchMatrix<int>(4, 5, 7, 0, 1);
        auto rm2 = Scalar<int>(13);
        auto mul = rm2 * rm1;
        auto mul_r = Evaluate(mul);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                for (size_t j = 0; j<5; ++j)
                {
                    assert(mul_r[b](i, j) == rm1[b](i, j) * 13);
                }
            }
        }
    }
    cout << "done" << endl;
}
}

void test_element_mul()
{
    test_mul1();
    test_mul2();
    test_mul3();
    test_mul4();
    test_mul5();
}
