#include <data_gen.h>
#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_dot_case1()
    {
        cout << "Test dot case 1 (matrix)\t";
        auto in1 = GenMatrix<CheckElement>(5, 3);
        auto in2 = GenMatrix<CheckElement>(3, 8, -10, 0.1);
        auto op = Dot(in1, in2);
        static_assert(IsMatrix<decltype(op)>);
        assert(op.Shape().RowNum() == 5);
        assert(op.Shape().ColNum() == 8);
        
        auto res = Evaluate(op);
        static_assert(IsMatrix<decltype(res)>);
        assert(res.Shape().RowNum() == 5);
        assert(res.Shape().ColNum() == 8);
        
        for (size_t i = 0; i < 5; ++i)
        {
            for (size_t j = 0; j < 8; ++j)
            {
                CheckElement value = 0;
                for (size_t k = 0; k < 3; ++k)
                {
                    value += in1(i, k) * in2(k, j);
                }
                assert(fabs(value - res(i, j)) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
    
    void test_dot_case2()
    {
        cout << "Test dot case 2 (batch matrix)\t";
        auto in1 = GenBatchMatrix<CheckElement>(7, 5, 3);
        auto in2 = GenBatchMatrix<CheckElement>(7, 3, 8, -10, 0.1);
        auto op = Dot(in1, in2);
        static_assert(IsBatchMatrix<decltype(op)>);
        assert(op.Shape().BatchNum() == 7);
        assert(op.Shape().RowNum() == 5);
        assert(op.Shape().ColNum() == 8);
        
        auto res = Evaluate(op);
        static_assert(IsBatchMatrix<decltype(res)>);
        assert(res.Shape().BatchNum() == 7);
        assert(res.Shape().RowNum() == 5);
        assert(res.Shape().ColNum() == 8);
        
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 5; ++i)
            {
                for (size_t j = 0; j < 8; ++j)
                {
                    CheckElement value = 0;
                    for (size_t k = 0; k < 3; ++k)
                    {
                        value += in1[b](i, k) * in2[b](k, j);
                    }
                    assert(fabs(value - res[b](i, j)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_dot_case3()
    {
        cout << "Test dot case 3 (matrix sequence)\t";
        auto in1 = GenMatrixSequence<CheckElement>(7, 5, 3);
        auto in2 = GenMatrixSequence<CheckElement>(7, 3, 8, -10, 0.1);
        auto op = Dot(in1, in2);
        static_assert(IsMatrixSequence<decltype(op)>);
        assert(op.Shape().Length() == 7);
        assert(op.Shape().RowNum() == 5);
        assert(op.Shape().ColNum() == 8);
        
        auto res = Evaluate(op);
        static_assert(IsMatrixSequence<decltype(res)>);
        assert(res.Shape().Length() == 7);
        assert(res.Shape().RowNum() == 5);
        assert(res.Shape().ColNum() == 8);
        
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t i = 0; i < 5; ++i)
            {
                for (size_t j = 0; j < 8; ++j)
                {
                    CheckElement value = 0;
                    for (size_t k = 0; k < 3; ++k)
                    {
                        value += in1[b](i, k) * in2[b](k, j);
                    }
                    assert(fabs(value - res[b](i, j)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_dot_case4()
    {
        cout << "Test dot case 4 (batch matrix sequence)\t";
        auto in1 = GenBatchMatrixSequence<CheckElement>(std::vector{3, 7, 2}, 5, 3);
        auto in2 = GenBatchMatrixSequence<CheckElement>(std::vector{3, 7, 2}, 3, 8, -10, 0.1);
        auto op = Dot(in1, in2);
        static_assert(IsBatchMatrixSequence<decltype(op)>);
        assert(op.Shape().SeqLenContainer().size() == 3);
        assert(op.Shape().SeqLenContainer()[0] == 3);
        assert(op.Shape().SeqLenContainer()[1] == 7);
        assert(op.Shape().SeqLenContainer()[2] == 2);
        assert(op.Shape().RowNum() == 5);
        assert(op.Shape().ColNum() == 8);
        
        auto res = Evaluate(op);
        static_assert(IsBatchMatrixSequence<decltype(res)>);
        assert(res.Shape().SeqLenContainer().size() == 3);
        assert(res.Shape().SeqLenContainer()[0] == 3);
        assert(res.Shape().SeqLenContainer()[1] == 7);
        assert(res.Shape().SeqLenContainer()[2] == 2);
        assert(res.Shape().RowNum() == 5);
        assert(res.Shape().ColNum() == 8);
        
        for (size_t i = 0; i < 5; ++i)
        {
            for (size_t j = 0; j < 8; ++j)
            {        
                for (size_t b = 0; b < 3; ++b)
                {
                    CheckElement value = 0;
                    for (size_t k = 0; k < 3; ++k)
                    {
                        value += in1[0][b](i, k) * in2[0][b](k, j);
                    }
                    assert(fabs(value - res[0][b](i, j)) < 0.001f);
                }
                for (size_t b = 0; b < 7; ++b)
                {
                    CheckElement value = 0;
                    for (size_t k = 0; k < 3; ++k)
                    {
                        value += in1[1][b](i, k) * in2[1][b](k, j);
                    }
                    assert(fabs(value - res[1][b](i, j)) < 0.001f);
                }
                for (size_t b = 0; b < 2; ++b)
                {
                    CheckElement value = 0;
                    for (size_t k = 0; k < 3; ++k)
                    {
                        value += in1[2][b](i, k) * in2[2][b](k, j);
                    }
                    assert(fabs(value - res[2][b](i, j)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators::Blas
{
    void test_dot()
    {
        test_dot_case1();
        test_dot_case2();
        test_dot_case3();
        test_dot_case4();
    }
}