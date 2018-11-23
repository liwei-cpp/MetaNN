#include <data/sequence/test_static_sequence.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_scalar_sequence_case1()
    {
        cout << "Test static scalar sequence case 1...\t";
        static_assert(IsScalarSequence<ScalarSequence<CheckElement, CheckDevice>>);
        static_assert(IsScalarSequence<ScalarSequence<CheckElement, CheckDevice>&>);
        static_assert(IsScalarSequence<ScalarSequence<CheckElement, CheckDevice>&&>);
        static_assert(IsScalarSequence<const ScalarSequence<CheckElement, CheckDevice>&>);
        static_assert(IsScalarSequence<const ScalarSequence<CheckElement, CheckDevice>&&>);

        ScalarSequence<CheckElement, CheckDevice> check;
        assert(check.Shape().Length() == 0);

        check = ScalarSequence<CheckElement, CheckDevice>::CreateWithShape(13);
        assert(check.Shape().Length() == 13);

        int c = 0;
        for (size_t i=0; i<13; ++i)
        {
            check.SetValue((float)(c++), i);
        }

        const ScalarSequence<CheckElement, CheckDevice> c2 = check;
        c = 0;
        for (size_t i=0; i<13; ++i)
        {
            assert(c2[i] == (float)(c++));
        }

        auto evalHandle = check.EvalRegister();
        auto cm = evalHandle.Data();

        for (size_t i = 0; i < cm.Shape().Length(); ++i)
        {
            assert(cm[i] == check[i]);
        }
        cout << "done" << endl;
    }
    
    void test_matrix_sequence_case1()
    {
        cout << "Test static matrix sequence case 1...\t";
        static_assert(IsMatrixSequence<MatrixSequence<int, CheckDevice>>);
        static_assert(IsMatrixSequence<MatrixSequence<int, CheckDevice> &>);
        static_assert(IsMatrixSequence<MatrixSequence<int, CheckDevice> &&>);
        static_assert(IsMatrixSequence<const MatrixSequence<int, CheckDevice> &>);
        static_assert(IsMatrixSequence<const MatrixSequence<int, CheckDevice> &&>);
    
        auto data = MatrixSequence<int, CheckDevice>::CreateWithShape(10, 13, 35);
        assert(data.AvailableForWrite());
        assert(data.Shape().Length() == 10);
        assert(data.Shape().RowNum() == 13);
        assert(data.Shape().ColNum() == 35);
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 13; ++j)
            {
                for (size_t k = 0; k < 35; ++k)
                {
                    data.SetValue((int)(i * 1000 + j * 100 + k), i, j, k);
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
    
    void test_matrix_sequence_case2()
    {
        cout << "Test static matrix sequence case 2...\t";
        
        auto rm1 = MatrixSequence<CheckElement, CheckDevice>::CreateWithShape(3, 10, 20);
        assert(rm1.Shape().Length() == 3);

        int c = 0;
        auto me1 = Matrix<CheckElement, CheckDevice>::CreateWithShape(10, 20);
        auto me2 = Matrix<CheckElement, CheckDevice>::CreateWithShape(10, 20);
        auto me3 = Matrix<CheckElement, CheckDevice>::CreateWithShape(10, 20);
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 20; ++j)
            {
                me1.SetValue((float)(c++), i, j);
                me2.SetValue((float)(c++), i, j);
                me3.SetValue((float)(c++), i, j);
                rm1.SetValue(me1(i, j), 0, i, j);
                rm1.SetValue(me2(i, j), 1, i, j);
                rm1.SetValue(me3(i, j), 2, i, j);
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
    
    void test_3d_array_sequence_case1()
    {
        cout << "Test static 3d array sequence case 1...\t";
        static_assert(IsThreeDArraySequence<ThreeDArraySequence<int, CheckDevice>>);
        static_assert(IsThreeDArraySequence<ThreeDArraySequence<int, CheckDevice> &>);
        static_assert(IsThreeDArraySequence<ThreeDArraySequence<int, CheckDevice> &&>);
        static_assert(IsThreeDArraySequence<const ThreeDArraySequence<int, CheckDevice> &>);
        static_assert(IsThreeDArraySequence<const ThreeDArraySequence<int, CheckDevice> &&>);
    
        auto data = ThreeDArraySequence<int, CheckDevice>::CreateWithShape(10, 7, 13, 35);
        assert(data.AvailableForWrite());
        assert(data.Shape().Length() == 10);
        assert(data.Shape().PageNum() == 7);
        assert(data.Shape().RowNum() == 13);
        assert(data.Shape().ColNum() == 35);
    
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t p = 0; p < 7; ++p)
            {
                for (size_t j = 0; j < 13; ++j)
                {
                    for (size_t k = 0; k < 35; ++k)
                    {
                        data.SetValue((int)(p * 33 + i * 1000 + j * 100 + k), i, p, j, k);
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

namespace Test::Data::Sequence
{
    void test_static_sequence()
    {
        test_scalar_sequence_case1();
        test_matrix_sequence_case1();
        test_matrix_sequence_case2();
        test_3d_array_sequence_case1();
    }
}