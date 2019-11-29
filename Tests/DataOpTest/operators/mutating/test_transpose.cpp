#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_transpose_case1()
    {
        cout << "Test transpose case 1 (matrix)\t";
        auto ori = GenMatrix<CheckElement>(10, 7, -1, 0.01);
        auto op = Transpose(ori);
        static_assert(IsMatrix<decltype(op)>);
        assert(op.Shape().RowNum() == 7);
        assert(op.Shape().ColNum() == 10);
        
        auto res = Evaluate(op);
        static_assert(IsMatrix<decltype(res)>);
        assert(res.Shape().RowNum() == 7);
        assert(res.Shape().ColNum() == 10);
        
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res(k, i) - ori(i, k)) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
    
    void test_transpose_case2()
    {
        cout << "Test transpose case 2 (batch matrix)\t";
        auto ori = GenBatchMatrix<CheckElement>(3, 10, 7, -1, 0.01);
        auto op = Transpose(ori);
        static_assert(IsBatchMatrix<decltype(op)>);
        assert(op.Shape().BatchNum() == 3);
        assert(op.Shape().RowNum() == 7);
        assert(op.Shape().ColNum() == 10);
        
        auto res = Evaluate(op);
        static_assert(IsBatchMatrix<decltype(res)>);
        assert(res.Shape().BatchNum() == 3);
        assert(res.Shape().RowNum() == 7);
        assert(res.Shape().ColNum() == 10);
        
        for (size_t b = 0; b < 3; ++b)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    assert(fabs(res[b](k, i) - ori[b](i, k)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_transpose_case3()
    {
        cout << "Test transpose case 3 (matrix sequence)\t";
        auto ori = GenMatrixSequence<CheckElement>(3, 10, 7, -1, 0.01);
        auto op = Transpose(ori);
        static_assert(IsMatrixSequence<decltype(op)>);
        assert(op.Shape().Length() == 3);
        assert(op.Shape().RowNum() == 7);
        assert(op.Shape().ColNum() == 10);
        
        auto res = Evaluate(op);
        static_assert(IsMatrixSequence<decltype(res)>);
        assert(res.Shape().Length() == 3);
        assert(res.Shape().RowNum() == 7);
        assert(res.Shape().ColNum() == 10);
        
        for (size_t b = 0; b < 3; ++b)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    assert(fabs(res[b](k, i) - ori[b](i, k)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_transpose_case4()
    {
        cout << "Test transpose case 4 (batch matrix sequence)\t";
        auto ori = GenBatchMatrixSequence<CheckElement>(std::vector{3, 5, 7}, 10, 7, -1, 0.01);
        auto op = Transpose(ori);
        static_assert(IsBatchMatrixSequence<decltype(op)>);
        assert(op.Shape().SeqLenContainer().size() == 3);
        assert(op.Shape().SeqLenContainer()[0] == 3);
        assert(op.Shape().SeqLenContainer()[1] == 5);
        assert(op.Shape().SeqLenContainer()[2] == 7);
        assert(op.Shape().RowNum() == 7);
        assert(op.Shape().ColNum() == 10);
        
        auto res = Evaluate(op);
        static_assert(IsBatchMatrixSequence<decltype(res)>);
        assert(res.Shape().SeqLenContainer().size() == 3);
        assert(res.Shape().SeqLenContainer()[0] == 3);
        assert(res.Shape().SeqLenContainer()[1] == 5);
        assert(res.Shape().SeqLenContainer()[2] == 7);
        assert(res.Shape().RowNum() == 7);
        assert(res.Shape().ColNum() == 10);

        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                for (size_t p = 0; p < 3; ++p)
                {
                    assert(fabs(res[0][p](k, i) - ori[0][p](i, k)) < 0.001f);
                }
                for (size_t p = 0; p < 5; ++p)
                {
                    assert(fabs(res[1][p](k, i) - ori[1][p](i, k)) < 0.001f);
                }
                for (size_t p = 0; p < 7; ++p)
                {
                    assert(fabs(res[2][p](k, i) - ori[2][p](i, k)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators::Mutating
{
    void test_transpose()
    {
        test_transpose_case1();
        test_transpose_case2();
        test_transpose_case3();
        test_transpose_case4();
    }
}