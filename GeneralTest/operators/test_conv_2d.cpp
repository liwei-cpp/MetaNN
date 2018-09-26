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

void test_conv_2d_case4()
{
    cout << "Test Conv 2D case 4 ...\t";
    auto input = GenThreeDArray<int>(1, 3, 3);
    auto kernel = GenSequenceThreeDArray<int>(2, 1, 2, 2);
    
    auto pad = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                    .template Set<ConvParams::RowNum>(0)
                    .template Set<ConvParams::ColNum>(0);
    auto strides = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                        .template Set<ConvParams::RowNum>(1)
                        .template Set<ConvParams::ColNum>(1);
    
    auto res = DefaultConv(input, kernel, pad, pad, strides);
    assert(res.PageNum() == 2);
    assert(res.RowNum() == 2);
    assert(res.ColNum() == 2);
    
    auto eval = Evaluate(res);
    assert(eval(0, 0, 0) == 19);
    assert(eval(0, 0, 1) == 25);
    assert(eval(0, 1, 0) == 37);
    assert(eval(0, 1, 1) == 43);
    assert(eval(1, 0, 0) == 51);
    assert(eval(1, 0, 1) == 73);
    assert(eval(1, 1, 0) == 117);
    assert(eval(1, 1, 1) == 139);
    
    cout << "done" << endl;
}

void test_conv_2d_case5()
{
    cout << "Test Conv 2D case 5 ...\t";
    auto input = GenThreeDArray<int>(1, 3, 3);
    auto kernel = GenSequenceThreeDArray<int>(2, 1, 2, 2);
    
    auto strides = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                        .template Set<ConvParams::RowNum>(1)
                        .template Set<ConvParams::ColNum>(1);
    
    auto res = SameConv(input, kernel, strides);
    assert(res.PageNum() == 2);
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
    
    assert(eval(1, 0, 0) == 51);
    assert(eval(1, 0, 1) == 73);
    assert(eval(1, 0, 2) == 38);
    assert(eval(1, 1, 0) == 117);
    assert(eval(1, 1, 1) == 139);
    assert(eval(1, 1, 2) == 68);
    assert(eval(1, 2, 0) == 59);
    assert(eval(1, 2, 1) == 68);
    assert(eval(1, 2, 2) == 32);

    cout << "done" << endl;
}

void test_conv_2d_case6()
{
    cout << "Test Conv 2D case 6 ...\t";
    auto input = GenThreeDArray<int>(1, 3, 3);
    auto kernel = GenSequenceThreeDArray<int>(2, 1, 2, 2);
    
    auto pad = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                    .template Set<ConvParams::RowNum>(1)
                    .template Set<ConvParams::ColNum>(1);
    auto strides = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                        .template Set<ConvParams::RowNum>(1)
                        .template Set<ConvParams::ColNum>(1);
    
    auto res = DefaultConv(input, kernel, pad, pad, strides);
    assert(res.PageNum() == 2);
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
    
    assert(eval(1, 0, 0) == 0);
    assert(eval(1, 0, 1) == 7);
    assert(eval(1, 0, 2) == 20);
    assert(eval(1, 0, 3) == 12);
    
    assert(eval(1, 1, 0) == 21);
    assert(eval(1, 1, 1) == 51);
    assert(eval(1, 1, 2) == 73);
    assert(eval(1, 1, 3) == 38);
    
    assert(eval(1, 2, 0) == 57);
    assert(eval(1, 2, 1) == 117);
    assert(eval(1, 2, 2) == 139);
    assert(eval(1, 2, 3) == 68);
    
    assert(eval(1, 3, 0) == 30);
    assert(eval(1, 3, 1) == 59);
    assert(eval(1, 3, 2) == 68);
    assert(eval(1, 3, 3) == 32);
    
    cout << "done" << endl;
}

void test_conv_2d_case7()
{
    cout << "Test Conv 2D case 7 ...\t";
    auto input = GenThreeDArray<int>(3, 3, 3);
    auto kernel = GenSequenceThreeDArray<int>(2, 3, 2, 2);
    
    auto pad = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                    .template Set<ConvParams::RowNum>(0)
                    .template Set<ConvParams::ColNum>(0);
    auto strides = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                        .template Set<ConvParams::RowNum>(1)
                        .template Set<ConvParams::ColNum>(1);
    
    auto res = DefaultConv(input, kernel, pad, pad, strides);
    assert(res.PageNum() == 2);
    assert(res.RowNum() == 2);
    assert(res.ColNum() == 2);
    
    auto eval = Evaluate(res);
    assert(eval(0, 0, 0) == 1035);
    assert(eval(0, 0, 1) == 1101);
    assert(eval(0, 1, 0) == 1233);
    assert(eval(0, 1, 1) == 1299);
    
    assert(eval(1, 0, 0) == 2619);
    assert(eval(1, 0, 1) == 2829);
    assert(eval(1, 1, 0) == 3249);
    assert(eval(1, 1, 1) == 3459);
    
    cout << "done" << endl;
}

void test_conv_2d_case8()
{
    cout << "Test Conv 2D case 8 ...\t";
    auto input = GenThreeDArray<int>(1, 3, 3);
    auto kernel = GenSequenceThreeDArray<int>(1, 1, 2, 2);
    
    auto pad = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                    .template Set<ConvParams::RowNum>(0)
                    .template Set<ConvParams::ColNum>(0);
    auto strides = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                        .template Set<ConvParams::RowNum>(2)
                        .template Set<ConvParams::ColNum>(2);
    
    auto res = DefaultConv(input, kernel, pad, pad, strides);
    assert(res.PageNum() == 1);
    assert(res.RowNum() == 1);
    assert(res.ColNum() == 1);
    
    auto eval = Evaluate(res);
    assert(eval(0, 0, 0) == 19);
    
    cout << "done" << endl;
}

void test_conv_2d_case9()
{
    cout << "Test Conv 2D case 9 ...\t";
    auto input = GenThreeDArray<int>(1, 3, 3);
    auto kernel = GenSequenceThreeDArray<int>(1, 1, 2, 2);
    
    auto padHead = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                    .template Set<ConvParams::RowNum>(0)
                    .template Set<ConvParams::ColNum>(0);
    auto padTail = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                    .template Set<ConvParams::RowNum>(1)
                    .template Set<ConvParams::ColNum>(1);
    auto strides = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                        .template Set<ConvParams::RowNum>(2)
                        .template Set<ConvParams::ColNum>(2);
    
    auto res = DefaultConv(input, kernel, padHead, padTail, strides);
    assert(res.PageNum() == 1);
    assert(res.RowNum() == 2);
    assert(res.ColNum() == 2);
    
    auto eval = Evaluate(res);
    assert(eval(0, 0, 0) == 19);
    assert(eval(0, 0, 1) == 10);
    assert(eval(0, 1, 0) == 7);
    assert(eval(0, 1, 1) == 0);
    
    cout << "done" << endl;
}

void test_conv_2d_case10()
{
    cout << "Test Conv 2D case 10 ...\t";
    auto input = GenThreeDArray<int>(3, 3, 3);
    auto kernel = GenSequenceThreeDArray<int>(2, 3, 2, 2);
    
    auto padHead = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                    .template Set<ConvParams::RowNum>(0)
                    .template Set<ConvParams::ColNum>(0);
    auto padTail = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                    .template Set<ConvParams::RowNum>(1)
                    .template Set<ConvParams::ColNum>(1);
    auto strides = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                        .template Set<ConvParams::RowNum>(2)
                        .template Set<ConvParams::ColNum>(2);
    
    auto res = DefaultConv(input, kernel, padHead, padTail, strides);
    assert(res.PageNum() == 2);
    assert(res.RowNum() == 2);
    assert(res.ColNum() == 2);
    
    auto eval = Evaluate(res);
    assert(eval(0, 0, 0) == 1035);
    assert(eval(0, 0, 1) == 528);
    assert(eval(0, 1, 0) == 564);
    assert(eval(0, 1, 1) == 276);

    assert(eval(1, 0, 0) == 2619);
    assert(eval(1, 0, 1) == 1428);
    assert(eval(1, 1, 0) == 1680);
    assert(eval(1, 1, 1) == 888);
    cout << "done" << endl;
}

void test_conv_2d_case11()
{
    cout << "Test Conv 2D case 11 ...\t";
    auto input = GenThreeDArray<int>(1, 3, 3);
    auto kernel = GenSequenceThreeDArray<int>(1, 1, 2, 2);
    
    auto strides = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                        .template Set<ConvParams::RowNum>(2)
                        .template Set<ConvParams::ColNum>(2);
    
    auto res = SameConv(input, kernel, strides);
    assert(res.PageNum() == 1);
    assert(res.RowNum() == 2);
    assert(res.ColNum() == 2);
    
    auto eval = Evaluate(res);
    assert(eval(0, 0, 0) == 19);
    assert(eval(0, 0, 1) == 10);
    assert(eval(0, 1, 0) == 7);
    assert(eval(0, 1, 1) == 0);
    
    cout << "done" << endl;
}
}

void test_conv_2d()
{
    // single channel, single kernel
    test_conv_2d_case1();
    test_conv_2d_case2();
    test_conv_2d_case3();
    
    // single channel, multiple kernel
    test_conv_2d_case4();
    test_conv_2d_case5();
    test_conv_2d_case6();
    
    // multiple channel, multiple kernel
    test_conv_2d_case7();
    
    // increase strides
    test_conv_2d_case8();
    test_conv_2d_case9();
    test_conv_2d_case10();
    
    // abnormal cases -- same behavior as caffe2
    test_conv_2d_case11();
}
