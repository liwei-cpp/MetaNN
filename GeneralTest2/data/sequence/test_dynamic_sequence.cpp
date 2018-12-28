#include <data/batch/test_dynamic_batch.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_dynamic_scalar_sequence_case1()
    {
        cout << "Test dynamic scalar sequence case 1...\t";
        using TCardinal = Scalar<CheckElement, CheckDevice>;
        
        static_assert(IsScalarSequence<DynamicSequence<TCardinal>>);
        static_assert(IsScalarSequence<DynamicSequence<TCardinal> &>);
        static_assert(IsScalarSequence<DynamicSequence<TCardinal> &&>);
        static_assert(IsScalarSequence<const DynamicSequence<TCardinal> &>);
        static_assert(IsScalarSequence<const DynamicSequence<TCardinal> &&>);

        auto rm1 = DynamicSequence<TCardinal>();
        assert(rm1.Shape().Length() == 0);
        assert(rm1.IsEmpty());

        rm1.PushBack(3);
        rm1.PushBack(8);
        rm1.PushBack(2);
        assert(rm1.Shape().Length() == 3);
        assert(!rm1.IsEmpty());

        auto evalHandle = rm1.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();
        auto rm2 = evalHandle.Data();
    
        assert(rm2[0] == 3);
        assert(rm2[1] == 8);
        assert(rm2[2] == 2);
        cout << "done" << endl;
    }
    
    void test_dynamic_matrix_sequence_case1()
    {
        cout << "Test dynamic matrix sequence case 1...\t";
        using TCardinal = Matrix<CheckElement, CheckDevice>;
        
        static_assert(IsMatrixSequence<DynamicSequence<TCardinal>>);
        static_assert(IsMatrixSequence<DynamicSequence<TCardinal> &>);
        static_assert(IsMatrixSequence<DynamicSequence<TCardinal> &&>);
        static_assert(IsMatrixSequence<const DynamicSequence<TCardinal> &>);
        static_assert(IsMatrixSequence<const DynamicSequence<TCardinal> &&>);

        DynamicSequence<TCardinal> rm1(10, 20);
        assert(rm1.Shape().Length() == 0);
        assert(rm1.IsEmpty());

        int c = 0;
        Matrix<CheckElement, CheckDevice> me1(10, 20);
        Matrix<CheckElement, CheckDevice> me2(10, 20);
        Matrix<CheckElement, CheckDevice> me3(10, 20);
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 20; ++j)
            {
                me1.SetValue((CheckElement)(c++), i, j);
                me2.SetValue((CheckElement)(c++), i, j);
                me3.SetValue((CheckElement)(c++), i, j);
            }
        }
        rm1.PushBack(me1);
        rm1.PushBack(me2);
        rm1.PushBack(me3);
        assert(rm1.Shape().Length() == 3);
        assert(!rm1.IsEmpty());

        auto evalHandle = rm1.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();
        auto rm2 = evalHandle.Data();

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
    
    void test_dynamic_3d_array_sequence_case1()
    {
        cout << "Test dynamic 3d array sequence case 1...\t";
        using TCardinal = ThreeDArray<CheckElement, CheckDevice>;
        
        static_assert(IsThreeDArraySequence<DynamicSequence<TCardinal>>);
        static_assert(IsThreeDArraySequence<DynamicSequence<TCardinal> &>);
        static_assert(IsThreeDArraySequence<DynamicSequence<TCardinal> &&>);
        static_assert(IsThreeDArraySequence<const DynamicSequence<TCardinal> &>);
        static_assert(IsThreeDArraySequence<const DynamicSequence<TCardinal> &&>);

        DynamicSequence<TCardinal> rm1(7, 10, 20);
        assert(rm1.Shape().Length() == 0);
        assert(rm1.IsEmpty());

        int c = 0;
        TCardinal me1(7, 10, 20);
        TCardinal me2(7, 10, 20);
        TCardinal me3(7, 10, 20);
        
        for (size_t p = 0; p < 7; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 20; ++j)
                {
                    me1.SetValue((CheckElement)(c++), p, i, j);
                    me2.SetValue((CheckElement)(c++), p, i, j);
                    me3.SetValue((CheckElement)(c++), p, i, j);
                }
            }
        }
        rm1.PushBack(me1);
        rm1.PushBack(me2);
        rm1.PushBack(me3);
        assert(rm1.Shape().Length() == 3);
        assert(!rm1.IsEmpty());

        auto evalHandle = rm1.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();
        auto rm2 = evalHandle.Data();

        for (size_t p = 0; p < 7; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 20; ++j)
                {
                    assert(rm1[0](p, i, j) == me1(p, i, j));
                    assert(rm1[1](p, i, j) == me2(p, i, j));
                    assert(rm1[2](p, i, j) == me3(p, i, j));
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Data::Sequence
{
    void test_dynamic_sequence()
    {
        test_dynamic_scalar_sequence_case1();
        test_dynamic_matrix_sequence_case1();
        test_dynamic_3d_array_sequence_case1();
    }
}