#include "test_batch_matrix.h"
#include "../facilities/calculate_tags.h"
#include <iostream>
#include <cassert>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
void test_batch_matrix1()
{
    cout << "Test batch matrix case 1...\t";
    static_assert(IsBatchMatrix<Batch<int, CheckDevice, CategoryTags::Matrix>>, "Test Error");
    static_assert(IsBatchMatrix<Batch<int, CheckDevice, CategoryTags::Matrix> &>, "Test Error");
    static_assert(IsBatchMatrix<Batch<int, CheckDevice, CategoryTags::Matrix> &&>, "Test Error");
    static_assert(IsBatchMatrix<const Batch<int, CheckDevice, CategoryTags::Matrix> &>, "Test Error");
    static_assert(IsBatchMatrix<const Batch<int, CheckDevice, CategoryTags::Matrix> &&>, "Test Error");
    
    Batch<int, CheckDevice, CategoryTags::Matrix> data(10, 13, 35);
    assert(data.AvailableForWrite());
    assert(data.BatchNum() == 10);
    assert(data.RowNum() == 13);
    assert(data.ColNum() == 35);
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 13; ++j)
        {
            for (size_t k = 0; k < 35; ++k)
            {
                data.SetValue(i, j, k, (int)(i * 1000 + j * 100 + k));
            }
        }
    }
    
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 13; ++j)
        {
            for (size_t k = 0; k < 35; ++k)
            {
                assert(data[i](j, k) == (int)(i * 1000 + j * 100 + k));
            }
        }
    }
    
    auto data2 = data.SubBatchMatrix(3, 7, 11, 22);
    assert(!data.AvailableForWrite());
    assert(!data2.AvailableForWrite());
    assert(data2.BatchNum() == 10);
    assert(data2.RowNum() == 4);
    assert(data2.ColNum() == 11);
    
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 3; j < 7; ++j)
        {
            for (size_t k = 11; k < 22; ++k)
            {
                assert(data2[i](j - 3, k - 11) == (int)(i * 1000 + j * 100 + k));
            }
        }
    }
    cout << "done" << endl;
}

void test_batch_matrix2()
{
    cout << "Test batch matrix case 2...\t";
    static_assert(IsBatchMatrix<Batch<CheckElement, CheckDevice, CategoryTags::Matrix>>, "Test Error");
    static_assert(IsBatchMatrix<Batch<CheckElement, CheckDevice, CategoryTags::Matrix> &>, "Test Error");
    static_assert(IsBatchMatrix<Batch<CheckElement, CheckDevice, CategoryTags::Matrix> &&>, "Test Error");
    static_assert(IsBatchMatrix<const Batch<CheckElement, CheckDevice, CategoryTags::Matrix> &>, "Test Error");
    static_assert(IsBatchMatrix<const Batch<CheckElement, CheckDevice, CategoryTags::Matrix> &&>, "Test Error");

    auto rm1 = Batch<CheckElement, CheckDevice, CategoryTags::Matrix>(3, 10, 20);
    assert(rm1.BatchNum() == 3);

    int c = 0;
    auto me1 = Matrix<CheckElement, CheckDevice>(10, 20);
    auto me2 = Matrix<CheckElement, CheckDevice>(10, 20);
    auto me3 = Matrix<CheckElement, CheckDevice>(10, 20);
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
        {
            me1.SetValue(i, j, (float)(c++));
            me2.SetValue(i, j, (float)(c++));
            me3.SetValue(i, j, (float)(c++));
            rm1.SetValue(0, i, j, me1(i, j));
            rm1.SetValue(1, i, j, me2(i, j));
            rm1.SetValue(2, i, j, me3(i, j));
        }
    }
    
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
        {
            assert(rm1[0](i, j) == me1(i, j));
            assert(rm1[1](i, j) == me2(i, j));
            assert(rm1[2](i, j) == me3(i, j));
        }
    }

    rm1 = rm1.SubBatchMatrix(3, 7, 11, 16);
    assert(rm1.RowNum() == 4);
    assert(rm1.ColNum() == 5);
    assert(rm1.BatchNum() == 3);
    me1 = me1.SubMatrix(3, 7, 11, 16);
    me2 = me2.SubMatrix(3, 7, 11, 16);
    me3 = me3.SubMatrix(3, 7, 11, 16);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            assert(rm1[0](i, j) == me1(i, j));
            assert(rm1[1](i, j) == me2(i, j));
            assert(rm1[2](i, j) == me3(i, j));
        }
    }

    auto evalHandle = rm1.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();
    auto rm2 = evalHandle.Data();

    for (size_t k = 0; k < 3; ++k)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                assert(rm1[k](i, j) == rm2[k](i, j));
            }
        }
    }
    cout << "done" << endl;
}
}

void test_batch_matrix()
{
    test_batch_matrix1();
    test_batch_matrix2();
}
