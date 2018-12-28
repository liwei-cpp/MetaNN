#include <data/batch_sequence/test_static_batch_sequence.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_batch_scalar_sequence_case1()
    {
        cout << "Test static batch scalar sequence case 1...\t";
        static_assert(IsBatchScalarSequence<BatchScalarSequence<CheckElement, CheckDevice>>);
        static_assert(IsBatchScalarSequence<BatchScalarSequence<CheckElement, CheckDevice>&>);
        static_assert(IsBatchScalarSequence<BatchScalarSequence<CheckElement, CheckDevice>&&>);
        static_assert(IsBatchScalarSequence<const BatchScalarSequence<CheckElement, CheckDevice>&>);
        static_assert(IsBatchScalarSequence<const BatchScalarSequence<CheckElement, CheckDevice>&&>);

        BatchScalarSequence<CheckElement, CheckDevice> check;
        assert(check.Shape().SeqLenContainer().empty());

        // check contains 4 sequences, with length = 13, 1, 100, 87
        const std::vector<size_t> seqs = {13, 1, 100, 87};
        check = BatchScalarSequence<CheckElement, CheckDevice>(seqs);
        assert(check.Shape().SeqLenContainer()[0] == 13);
        assert(check.Shape().SeqLenContainer()[1] == 1);
        assert(check.Shape().SeqLenContainer()[2] == 100);
        assert(check.Shape().SeqLenContainer()[3] == 87);

        int c = 0;
        for (size_t s = 0; s < 4; ++s)
        {
            for (size_t i = 0; i < seqs[s]; ++i)
            {
                check.SetValue((float)(c++), s, i);
            }
        }

        const auto c2 = check;
        c = 0;
        for (size_t s = 0; s < 4; ++s)
        {
            for (size_t i = 0; i < seqs[s]; ++i)
            {
                assert(c2[s][i].Value() == (float)(c++));
            }
        }

        auto evalHandle = check.EvalRegister();
        auto cm = evalHandle.Data();

        const auto& seqCont = cm.Shape().SeqLenContainer();
        for (size_t i = 0; i < seqCont.size(); ++i)
        {
            for (size_t j = 0; j < seqCont[i]; ++j)
            {
                assert(cm[i][j] == check[i][j]);
            }
        }
        cout << "done" << endl;
    }
    
    void test_batch_matrix_sequence_case1()
    {
        cout << "Test static batch matrix sequence case 1...\t";
        static_assert(IsBatchMatrixSequence<BatchMatrixSequence<int, CheckDevice>>);
        static_assert(IsBatchMatrixSequence<BatchMatrixSequence<int, CheckDevice> &>);
        static_assert(IsBatchMatrixSequence<BatchMatrixSequence<int, CheckDevice> &&>);
        static_assert(IsBatchMatrixSequence<const BatchMatrixSequence<int, CheckDevice> &>);
        static_assert(IsBatchMatrixSequence<const BatchMatrixSequence<int, CheckDevice> &&>);
    
        // check contains 4 sequences, with length = 13, 1, 100, 87
        const std::vector<size_t> seqs = {13, 1, 100, 87};
        BatchMatrixSequence<int, CheckDevice> data(seqs, 13, 35);
        assert(data.AvailableForWrite());
        assert(data.Shape().SeqLenContainer()[0] == 13);
        assert(data.Shape().SeqLenContainer()[1] == 1);
        assert(data.Shape().SeqLenContainer()[2] == 100);
        assert(data.Shape().SeqLenContainer()[3] == 87);
        assert(data.Shape().RowNum() == 13);
        assert(data.Shape().ColNum() == 35);

        for (size_t s = 0; s < seqs.size(); ++s)
        {
            for (size_t i = 0; i < seqs[s]; ++i)
            {
                for (size_t j = 0; j < 13; ++j)
                {
                    for (size_t k = 0; k < 35; ++k)
                    {
                        data.SetValue((int)(s * 10000 + i * 1000 + j * 100 + k), s, i, j, k);
                    }
                }
            }
        }

        for (size_t s = 0; s < seqs.size(); ++s)
        {
            for (size_t i = 0; i < seqs[s]; ++i)
            {
                for (size_t j = 0; j < 13; ++j)
                {
                    for (size_t k = 0; k < 35; ++k)
                    {
                        assert(data[s][i](j, k) == (int)(s * 10000 + i * 1000 + j * 100 + k));
                    }
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_batch_matrix_sequence_case2()
    {
        cout << "Test static batch matrix sequence case 2...\t";

        const std::vector<size_t> seqs = {13, 1, 100};
        BatchMatrixSequence<CheckElement, CheckDevice> rm1(seqs, 10, 20);
        assert(rm1.Shape().SeqLenContainer()[0] == 13);
        assert(rm1.Shape().SeqLenContainer()[1] == 1);
        assert(rm1.Shape().SeqLenContainer()[2] == 100);

        MatrixSequence<CheckElement, CheckDevice> me1(13, 10, 20);
        MatrixSequence<CheckElement, CheckDevice> me2(1, 10, 20);
        MatrixSequence<CheckElement, CheckDevice> me3(100, 10, 20);
        
        int c = 0;
        for (size_t len = 0; len < 13; ++len)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 20; ++j)
                {
                    me1.SetValue((float)(c++), len, i, j);
                    rm1.SetValue(me1[len](i, j), 0, len, i, j);
                }
            }
        }
        for (size_t len = 0; len < 1; ++len)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 20; ++j)
                {
                    me2.SetValue((float)(c++), len, i, j);
                    rm1.SetValue(me2[len](i, j), 1, len, i, j);
                }
            }
        }
        for (size_t len = 0; len < 100; ++len)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 20; ++j)
                {
                    me3.SetValue((float)(c++), len, i, j);
                    rm1.SetValue(me3[len](i, j), 2, len, i, j);
                }
            }
        }
        
        for (size_t len = 0; len < 13; ++len)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 20; ++j)
                {
                    assert(rm1[0][len](i, j) == me1[len](i, j));
                }
            }
        }
        for (size_t len = 0; len < 1; ++len)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 20; ++j)
                {
                    assert(rm1[1][len](i, j) == me2[len](i, j));
                }
            }
        }
        for (size_t len = 0; len < 100; ++len)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 20; ++j)
                {
                    assert(rm1[2][len](i, j) == me3[len](i, j));
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_batch_3d_array_sequence_case1()
    {
        cout << "Test static batch 3d array sequence case 1...\t";
        static_assert(IsBatchThreeDArraySequence<BatchThreeDArraySequence<int, CheckDevice>>);
        static_assert(IsBatchThreeDArraySequence<BatchThreeDArraySequence<int, CheckDevice> &>);
        static_assert(IsBatchThreeDArraySequence<BatchThreeDArraySequence<int, CheckDevice> &&>);
        static_assert(IsBatchThreeDArraySequence<const BatchThreeDArraySequence<int, CheckDevice> &>);
        static_assert(IsBatchThreeDArraySequence<const BatchThreeDArraySequence<int, CheckDevice> &&>);

        const std::vector<size_t> seqs = {13, 1, 100};
        BatchThreeDArraySequence<int, CheckDevice> data(seqs, 7, 13, 35);
        assert(data.AvailableForWrite());
        assert(data.Shape().SeqLenContainer()[0] == 13);
        assert(data.Shape().SeqLenContainer()[1] == 1);
        assert(data.Shape().SeqLenContainer()[2] == 100);
        assert(data.Shape().PageNum() == 7);
        assert(data.Shape().RowNum() == 13);
        assert(data.Shape().ColNum() == 35);

        for (size_t s = 0; s< seqs.size(); ++s)
        {
            for (size_t i = 0; i < seqs[s]; ++i)
            {
                for (size_t p = 0; p < 7; ++p)
                {
                    for (size_t j = 0; j < 13; ++j)
                    {
                        for (size_t k = 0; k < 35; ++k)
                        {
                            data.SetValue((int)(s * 171 + p * 33 + i * 1000 + j * 100 + k), s, i, p, j, k);
                        }
                    }
                }
            }
        }
        
        for (size_t s = 0; s< seqs.size(); ++s)
        {
            for (size_t i = 0; i < seqs[s]; ++i)
            {
                for (size_t p = 0; p < 7; ++p)
                {
                    for (size_t j = 0; j < 13; ++j)
                    {
                        for (size_t k = 0; k < 35; ++k)
                        {
                            assert(data[s][i](p, j, k) == (int)(s * 171 + p * 33 + i * 1000 + j * 100 + k));
                        }
                    }
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Data::BatchSequence
{
    void test_static_batch_sequence()
    {
        test_batch_scalar_sequence_case1();
        test_batch_matrix_sequence_case1();
        test_batch_matrix_sequence_case2();
        test_batch_3d_array_sequence_case1();
    }
}