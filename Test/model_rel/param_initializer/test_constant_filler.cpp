#include "test_constant_filler.h"
#include <MetaNN/meta_nn.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_constant_filler1()
{
    cout << "test constant filler case 1 ...";
    
    ConstantFiller filler(0);
    Matrix<float, DeviceTags::CPU> mat (11, 13);
    filler.Fill(mat, 11, 13);
    for (size_t i = 0; i < 11; ++i)
    {
        for (size_t j = 0; j < 13; ++j)
        {
            assert(fabs(mat(i, j)) < 0.0001);
        }
    }
    
    ConstantFiller filler2(1.5f);
    Matrix<float, DeviceTags::CPU> mat2 (21, 33);
    filler2.Fill(mat2, 21, 33);
    for (size_t i = 0; i < 21; ++i)
    {
        for (size_t j = 0; j < 33; ++j)
        {
            assert(fabs(mat2(i, j) - 1.5) < 0.0001);
        }
    }
    
    cout << "done" << endl;
}
}

void test_constant_filler()
{
    test_constant_filler1();
}
