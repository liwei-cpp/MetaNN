#include "test_batch_matrix.h"
#include "../facilities/calculate_tags.h"
#include <iostream>
#include <cassert>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
void test_sequence_matrix1()
{
    cout << "Test sequence matrix case 1...\t";
    static_assert(IsMatrixSequence<Sequence<int, CheckDevice, CategoryTags::Matrix>>, "Test Error");
    static_assert(IsMatrixSequence<Sequence<int, CheckDevice, CategoryTags::Matrix> &>, "Test Error");
    static_assert(IsMatrixSequence<Sequence<int, CheckDevice, CategoryTags::Matrix> &&>, "Test Error");
    static_assert(IsMatrixSequence<const Sequence<int, CheckDevice, CategoryTags::Matrix> &>, "Test Error");
    static_assert(IsMatrixSequence<const Sequence<int, CheckDevice, CategoryTags::Matrix> &&>, "Test Error");
    
    Sequence<int, CheckDevice, CategoryTags::Matrix> data(10, 13, 35);
    assert(data.AvailableForWrite());
    assert(data.Length() == 10);
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
    cout << "done" << endl;
}

void test_sequence_matrix2()
{
    cout << "Test sequence matrix case 2...\t";
    static_assert(IsMatrixSequence<Sequence<CheckElement, CheckDevice, CategoryTags::Matrix>>, "Test Error");
    static_assert(IsMatrixSequence<Sequence<CheckElement, CheckDevice, CategoryTags::Matrix> &>, "Test Error");
    static_assert(IsMatrixSequence<Sequence<CheckElement, CheckDevice, CategoryTags::Matrix> &&>, "Test Error");
    static_assert(IsMatrixSequence<const Sequence<CheckElement, CheckDevice, CategoryTags::Matrix> &>, "Test Error");
    static_assert(IsMatrixSequence<const Sequence<CheckElement, CheckDevice, CategoryTags::Matrix> &&>, "Test Error");

    auto rm1 = Sequence<CheckElement, CheckDevice, CategoryTags::Matrix>(3, 10, 20);
    assert(rm1.Length() == 3);

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
    cout << "done" << endl;
}
}

void test_sequence_matrix()
{
    test_sequence_matrix1();
    test_sequence_matrix2();
}
