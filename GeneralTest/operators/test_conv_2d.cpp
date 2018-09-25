#include "test_conv_2d.h"
#include "../facilities/data_gen.h"
#include <MetaNN/meta_nn.h>
#include <cmath>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_conv_2d_case1()
{
    cout << "Test Conv 2D case 1 ...\t";
    auto input = GenThreeDArray<int>(1, 3, 3);
    auto kernel = GenSequenceThreeDArray<int>(1, 1, 2, 2);
    
    auto pad = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                    .template Set<ConvParams::RowNum>(0)
                    .template Set<ConvParams::ColNum>(0);
    auto strides = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                        .template Set<ConvParams::RowNum>(1)
                        .template Set<ConvParams::ColNum>(1);
    
    auto res = DefaultConv(input, kernel, pad, pad, strides);
    assert(res.PageNum() == 1);
    assert(res.RowNum() == 2);
    assert(res.ColNum() == 2);
    
    auto eval = Evaluate(res);
    assert(eval(0, 0, 0) == 19);
    assert(eval(0, 0, 1) == 25);
    assert(eval(0, 1, 0) == 37);
    assert(eval(0, 1, 1) == 43);
    
    cout << "done" << endl;
}

void test_conv_2d_case2()
{
    cout << "Test Conv 2D case 2 ...\t";
    auto input = GenThreeDArray<int>(1, 3, 3);
    auto kernel = GenSequenceThreeDArray<int>(1, 1, 2, 2);
    
    auto strides = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                        .template Set<ConvParams::RowNum>(1)
                        .template Set<ConvParams::ColNum>(1);
    
    auto res = SameConv(input, kernel, strides);
    assert(res.PageNum() == 1);
    assert(res.RowNum() == 3);
    assert(res.ColNum() == 3);
    
    auto eval = Evaluate(res);
    assert(eval(0, 0, 0) == 19);
    assert(eval(0, 0, 1) == 25);
    assert(eval(0, 0, 2) == 10);
    
    assert(eval(0, 1, 0) == 37);
    assert(eval(0, 1, 1) == 43);
    assert(eval(0, 1, 2) == 16);
    
    assert(eval(0, 2, 0) == 7);
    assert(eval(0, 2, 1) == 8);
    assert(eval(0, 2, 2) == 0);

    cout << "done" << endl;
}

void test_conv_2d_case3()
{
    cout << "Test Conv 2D case 3 ...\t";
    auto input = GenThreeDArray<int>(1, 3, 3);
    auto kernel = GenSequenceThreeDArray<int>(1, 1, 2, 2);
    
    auto pad = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                    .template Set<ConvParams::RowNum>(1)
                    .template Set<ConvParams::ColNum>(1);
    auto strides = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                        .template Set<ConvParams::RowNum>(1)
                        .template Set<ConvParams::ColNum>(1);
    
    auto res = DefaultConv(input, kernel, pad, pad, strides);
    assert(res.PageNum() == 1);
    assert(res.RowNum() == 4);
    assert(res.ColNum() == 4);
    
    auto eval = Evaluate(res);
    assert(eval(0, 0, 0) == 0);
    assert(eval(0, 0, 1) == 3);
    assert(eval(0, 0, 2) == 8);
    assert(eval(0, 0, 3) == 4);
    
    assert(eval(0, 1, 0) == 9);
    assert(eval(0, 1, 1) == 19);
    assert(eval(0, 1, 2) == 25);
    assert(eval(0, 1, 3) == 10);
    
    assert(eval(0, 2, 0) == 21);
    assert(eval(0, 2, 1) == 37);
    assert(eval(0, 2, 2) == 43);
    assert(eval(0, 2, 3) == 16);
    
    assert(eval(0, 3, 0) == 6);
    assert(eval(0, 3, 1) == 7);
    assert(eval(0, 3, 2) == 8);
    assert(eval(0, 3, 3) == 0);
    
    cout << "done" << endl;
}
}

void test_conv_2d()
{
    test_conv_2d_case1();
    test_conv_2d_case2();
    test_conv_2d_case3();
    
    // TODO: multiple kernel
    // TODO: increase strides
}
