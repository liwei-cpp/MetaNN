#include "test_add.h"
#include "../facilities/calculate_tags.h"
#include "../facilities/data_gen.h"
#include <MetaNN/meta_nn.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_add1()
{
    cout << "Test add case 1 ...\t";
    auto rm1 = GenMatrix<int>(4, 5, 0, 1);
    auto rm2 = GenMatrix<int>(4, 5, 2, 3);
    auto add = rm1 + rm2;
    auto add_r = Evaluate(add);

    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(add_r(i, j) == rm1(i, j) + rm2(i, j));
        }
    }

    rm1 = GenMatrix<int>(111, 113, 1, 2);
    rm2 = GenMatrix<int>(111, 113, 2, 3);
    rm1 = rm1.SubMatrix(31, 35, 17, 22);
    rm2 = rm2.SubMatrix(41, 45, 27, 32);
    add = rm1 + rm2;
    add_r = Evaluate(add);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            assert(add_r(i, j) == rm1(i, j) + rm2(i, j));
        }
    }
    cout << "done" << endl;
}

void test_add2()
{
    cout << "Test add case 2 ...\t";
    auto rm1 = GenMatrix<int>(4, 5, 0, 1);
    auto add = rm1 + MetaNN::Scalar<int, DeviceTags::CPU>(2);
    auto add_r = Evaluate(add);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(add_r(i, j) == rm1(i, j) + 2);
        }
    }

    rm1 = GenMatrix<int>(111, 113, 2, 3);
    rm1 = rm1.SubMatrix(31, 35, 17, 22);
    add = Scalar<int>(3) + rm1;
    add_r = Evaluate(add);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            assert(add_r(i, j) == rm1(i, j) + 3);
        }
    }
    cout << "done" << endl;
}

void test_add3()
{
    cout << "Test add case 3 ...\t";
    auto rm1 = MakeTrivalMatrix<int, CheckDevice>(2, 10, 3);
    auto rm2 = MakeTrivalMatrix<int, CheckDevice>(2, 10, 5);
    auto add = rm1 + rm2;
    auto add_r = Evaluate(add);
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j<10; ++j)
        {
            assert(add_r(i, j) == 8);
        }
    }
    cout << "done" << endl;
}

void test_add4()
{
    cout << "Test add case 4 ...\t";
    auto rm1 = GenBatchMatrix<int>(4, 5, 7, 1, -1);
    auto rm2 = GenMatrix<int>(4, 5, 2, 3);
    auto add = rm1 + rm2;
    auto add_r = Evaluate(add);
    
    assert(add.RowNum() == 4);
    assert(add.ColNum() == 5);
    assert(add.BatchNum() == 7);

    for (size_t b = 0 ; b < 7; ++b)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                assert(add_r[b](i, j) == rm1[b](i, j) + rm2(i, j));
            }
        }
    }
    
    auto add2 = rm2 + rm1;
    add_r = Evaluate(add2);
    assert(add2.RowNum() == 4);
    assert(add2.ColNum() == 5);
    assert(add2.BatchNum() == 7);

    for (size_t b = 0 ; b < 7; ++b)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                assert(add_r[b](i, j) == rm1[b](i, j) + rm2(i, j));
            }
        }
    }
    cout << "done" << endl;
}

void test_add5()
{
    cout << "Test add case 5 ...\t";
    auto rm1 = GenBatchMatrix<int>(4, 5, 7, 1, -1);
    auto rm2 = GenBatchMatrix<int>(4, 5, 7, 2, 3);
    auto add = rm1 + rm2;
    auto add_r = Evaluate(add);
    
    assert(add.RowNum() == 4);
    assert(add.ColNum() == 5);
    assert(add.BatchNum() == 7);

    for (size_t b = 0 ; b < 7; ++b)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                assert(add_r[b](i, j) == rm1[b](i, j) + rm2[b](i, j));
            }
        }
    }
    cout << "done" << endl;
}

void test_add6()
{
    cout << "Test add case 6 ...\t";
    auto rm1 = GenBatchMatrix<int>(4, 5, 7, 1, -1);
    auto add = rm1 + Scalar<int>(3);
    auto add_r = Evaluate(add);
    
    assert(add.RowNum() == 4);
    assert(add.ColNum() == 5);
    assert(add.BatchNum() == 7);

    for (size_t b = 0 ; b < 7; ++b)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                assert(add_r[b](i, j) == rm1[b](i, j) + 3);
            }
        }
    }
    
    auto add2 = Scalar<int>(3) + rm1;
    add_r = Evaluate(add2);
    
    assert(add2.RowNum() == 4);
    assert(add2.ColNum() == 5);
    assert(add2.BatchNum() == 7);

    for (size_t b = 0 ; b < 7; ++b)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                assert(add_r[b](i, j) == rm1[b](i, j) + 3);
            }
        }
    }
    cout << "done" << endl;
}
}

void test_add()
{
    test_add1();
    test_add2();
    test_add3();
    test_add4();
    test_add5();
    test_add6();
}
