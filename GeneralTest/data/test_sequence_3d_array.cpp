#include "test_batch_matrix.h"
#include "../facilities/calculate_tags.h"
#include <iostream>
#include <cassert>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
void test_sequence_3d_array1()
{
    cout << "Test sequence 3d array case 1...\t";
    static_assert(IsThreeDArraySequence<Sequence<int, CheckDevice, CategoryTags::ThreeDArray>>);
    static_assert(IsThreeDArraySequence<Sequence<int, CheckDevice, CategoryTags::ThreeDArray> &>);
    static_assert(IsThreeDArraySequence<Sequence<int, CheckDevice, CategoryTags::ThreeDArray> &&>);
    static_assert(IsThreeDArraySequence<const Sequence<int, CheckDevice, CategoryTags::ThreeDArray> &>);
    static_assert(IsThreeDArraySequence<const Sequence<int, CheckDevice, CategoryTags::ThreeDArray> &&>);
    
    Sequence<int, CheckDevice, CategoryTags::ThreeDArray> data(10, 7, 13, 35);
    assert(data.AvailableForWrite());
    assert(data.Length() == 10);
    assert(data.PageNum() == 7);
    assert(data.RowNum() == 13);
    assert(data.ColNum() == 35);
    
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t p = 0; p < 7; ++p)
        {
            for (size_t j = 0; j < 13; ++j)
            {
                for (size_t k = 0; k < 35; ++k)
                {
                    data.SetValue(i, p, j, k, (int)(p * 33 + i * 1000 + j * 100 + k));
                }
            }
        }
    }
    
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t p = 0; p < 7; ++p)
        {
            for (size_t j = 0; j < 13; ++j)
            {
                for (size_t k = 0; k < 35; ++k)
                {
                    assert(data[i](p, j, k) == (int)(p * 33 + i * 1000 + j * 100 + k));
                }
            }
        }
    }
    cout << "done" << endl;
}
}

void test_sequence_3d_array()
{
    test_sequence_3d_array1();
}
