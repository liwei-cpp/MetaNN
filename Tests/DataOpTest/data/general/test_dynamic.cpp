#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_dynamic_scalar_case1()
    {
        cout << "Test dynamic scalar case 1...\t";
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::Scalar>;
        
        static_assert(IsDynamic<CheckType>);
        static_assert(IsScalar<CheckType>);
        
        Scalar<CheckElement, CheckDevice> internal;
        internal.SetValue(3);
        CheckType dScalar = MakeDynamic(std::move(internal));
        
        CheckType dScalar2;
        
        assert(dScalar == dScalar);
        assert(dScalar != dScalar2);
        
        assert(!dScalar.IsEmpty());
        assert(dScalar2.IsEmpty());
        
        assert(dScalar.Shape().Count() == 1);
        
        auto castPtr1 = dScalar.TryCastTo<Scalar<CheckElement, CheckDevice>>();
        assert(castPtr1);
        auto castPtr2 = dScalar2.TryCastTo<Scalar<CheckElement, CheckDevice>>();
        assert(!castPtr2);
        
        auto evalRes = Evaluate(dScalar);
        assert(evalRes.Value() == 3);
        cout << "done" << endl;
    }
    
    void test_dynamic_matrix_case1()
    {
        cout << "Test dynamic matrix case 1...\t";
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::Matrix>;
        
        static_assert(IsDynamic<CheckType>);
        static_assert(IsMatrix<CheckType>);
        
        Matrix<CheckElement, CheckDevice> internal(11, 13);
        CheckType dMatrix = MakeDynamic(internal);
        
        CheckType dMatrix2;
        
        assert(dMatrix == dMatrix);
        assert(dMatrix != dMatrix2);
        
        assert(!dMatrix.IsEmpty());
        assert(dMatrix2.IsEmpty());
        
        assert(dMatrix.Shape().RowNum() == 11);
        assert(dMatrix.Shape().ColNum() == 13);
        
        auto castPtr1 = dMatrix.TryCastTo<Matrix<CheckElement, CheckDevice>>();
        assert(castPtr1);
        auto castPtr2 = dMatrix2.TryCastTo<Matrix<CheckElement, CheckDevice>>();
        assert(!castPtr2);
        auto castPtr3 = dMatrix.TryCastTo<Scalar<CheckElement, CheckDevice>>();
        assert(!castPtr3);
        
        auto evalRes = Evaluate(dMatrix);
        assert(evalRes == internal);

        cout << "done" << endl;
    }
    
    void test_dynamic_3d_array_case1()
    {
        cout << "Test dynamic 3d array case 1...\t";
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::ThreeDArray>;
        
        static_assert(IsDynamic<CheckType>);
        static_assert(IsThreeDArray<CheckType>);
        
        ThreeDArray<CheckElement, CheckDevice> internal(7, 11, 13);
        CheckType dChecker = MakeDynamic(internal);
        
        CheckType dChecker2;
        
        assert(dChecker == dChecker);
        assert(dChecker != dChecker2);
        
        assert(!dChecker.IsEmpty());
        assert(dChecker2.IsEmpty());
        
        assert(dChecker.Shape().PageNum() == 7);
        assert(dChecker.Shape().RowNum() == 11);
        assert(dChecker.Shape().ColNum() == 13);
        
        auto castPtr1 = dChecker.TryCastTo<ThreeDArray<CheckElement, CheckDevice>>();
        assert(castPtr1);
        auto castPtr2 = dChecker2.TryCastTo<ThreeDArray<CheckElement, CheckDevice>>();
        assert(!castPtr2);
        auto castPtr3 = dChecker.TryCastTo<Scalar<CheckElement, CheckDevice>>();
        assert(!castPtr3);
        
        auto evalRes = Evaluate(dChecker);
        assert(evalRes == internal);

        cout << "done" << endl;
    }
    
    void test_dynamic_batch_scalar_case1()
    {
        cout << "Test dynamic batch scalar case 1...\t";
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::BatchScalar>;
        
        static_assert(IsDynamic<CheckType>);
        static_assert(IsBatchScalar<CheckType>);
        
        BatchScalar<CheckElement, CheckDevice> internal{11};
        for (size_t i = 0; i < 11; ++i)
        {
            internal.SetValue(i, i);
        }

        CheckType dScalar = MakeDynamic(internal);
        
        CheckType dScalar2;
        
        assert(dScalar == dScalar);
        assert(dScalar != dScalar2);
        
        assert(!dScalar.IsEmpty());
        assert(dScalar2.IsEmpty());
        
        assert(dScalar.Shape().BatchNum() == 11);
        
        auto castPtr1 = dScalar.TryCastTo<BatchScalar<CheckElement, CheckDevice>>();
        assert(castPtr1);
        auto castPtr2 = dScalar2.TryCastTo<BatchScalar<CheckElement, CheckDevice>>();
        assert(!castPtr2);
        
        auto evalRes = Evaluate(dScalar);
        assert(evalRes == internal);
        cout << "done" << endl;
    }
    
    void test_dynamic_batch_matrix_case1()
    {
        cout << "Test dynamic batch matrix case 1...\t";
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::BatchMatrix>;
        
        static_assert(IsDynamic<CheckType>);
        static_assert(IsBatchMatrix<CheckType>);
        
        BatchMatrix<CheckElement, CheckDevice> internal(7, 11, 13);
        CheckType dMatrix = MakeDynamic(internal);
        
        CheckType dMatrix2;
        
        assert(dMatrix == dMatrix);
        assert(dMatrix != dMatrix2);
        
        assert(!dMatrix.IsEmpty());
        assert(dMatrix2.IsEmpty());
        assert(dMatrix.Shape().BatchNum() == 7);
        assert(dMatrix.Shape().RowNum() == 11);
        assert(dMatrix.Shape().ColNum() == 13);
        
        auto castPtr1 = dMatrix.TryCastTo<BatchMatrix<CheckElement, CheckDevice>>();
        assert(castPtr1);
        auto castPtr2 = dMatrix2.TryCastTo<BatchMatrix<CheckElement, CheckDevice>>();
        assert(!castPtr2);
        auto castPtr3 = dMatrix.TryCastTo<Scalar<CheckElement, CheckDevice>>();
        assert(!castPtr3);
        
        auto evalRes = Evaluate(dMatrix);
        assert(evalRes == internal);

        cout << "done" << endl;
    }
    
    void test_dynamic_batch_3d_array_case1()
    {
        cout << "Test dynamic batch 3d array case 1...\t";
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::BatchThreeDArray>;
        
        static_assert(IsDynamic<CheckType>);
        static_assert(IsBatchThreeDArray<CheckType>);
        
        BatchThreeDArray<CheckElement, CheckDevice> internal(51, 7, 11, 13);
        CheckType dChecker = MakeDynamic(internal);
        
        CheckType dChecker2;
        
        assert(dChecker == dChecker);
        assert(dChecker != dChecker2);
        
        assert(!dChecker.IsEmpty());
        assert(dChecker2.IsEmpty());
        
        assert(dChecker.Shape().BatchNum() == 51);
        assert(dChecker.Shape().PageNum() == 7);
        assert(dChecker.Shape().RowNum() == 11);
        assert(dChecker.Shape().ColNum() == 13);
        
        auto castPtr1 = dChecker.TryCastTo<BatchThreeDArray<CheckElement, CheckDevice>>();
        assert(castPtr1);
        auto castPtr2 = dChecker2.TryCastTo<BatchThreeDArray<CheckElement, CheckDevice>>();
        assert(!castPtr2);
        auto castPtr3 = dChecker.TryCastTo<Scalar<CheckElement, CheckDevice>>();
        assert(!castPtr3);
        
        auto evalRes = Evaluate(dChecker);
        assert(evalRes == internal);

        cout << "done" << endl;
    }
    
    void test_dynamic_scalar_sequence_case1()
    {
        cout << "Test dynamic scalar sequence case 1...\t";
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::ScalarSequence>;
        
        static_assert(IsDynamic<CheckType>);
        static_assert(IsScalarSequence<CheckType>);
        
        ScalarSequence<CheckElement, CheckDevice> internal(11);
        for (size_t i = 0; i < 11; ++i)
        {
            internal.SetValue(i, i);
        }

        CheckType dScalar = MakeDynamic(internal);
        
        CheckType dScalar2;
        
        assert(dScalar == dScalar);
        assert(dScalar != dScalar2);
        
        assert(!dScalar.IsEmpty());
        assert(dScalar2.IsEmpty());
        
        assert(dScalar.Shape().Length() == 11);
        
        auto castPtr1 = dScalar.TryCastTo<ScalarSequence<CheckElement, CheckDevice>>();
        assert(castPtr1);
        auto castPtr2 = dScalar2.TryCastTo<ScalarSequence<CheckElement, CheckDevice>>();
        assert(!castPtr2);
        
        auto evalRes = Evaluate(dScalar);
        assert(evalRes == internal);
        cout << "done" << endl;
    }
    
    void test_dynamic_matrix_sequence_case1()
    {
        cout << "Test dynamic matrix sequence case 1...\t";
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::MatrixSequence>;
        
        static_assert(IsDynamic<CheckType>);
        static_assert(IsMatrixSequence<CheckType>);
        
        MatrixSequence<CheckElement, CheckDevice> internal(7, 11, 13);
        CheckType dMatrix = MakeDynamic(internal);
        
        CheckType dMatrix2;
        
        assert(dMatrix == dMatrix);
        assert(dMatrix != dMatrix2);
        
        assert(!dMatrix.IsEmpty());
        assert(dMatrix2.IsEmpty());
        assert(dMatrix.Shape().Length() == 7);
        assert(dMatrix.Shape().RowNum() == 11);
        assert(dMatrix.Shape().ColNum() == 13);
        
        auto castPtr1 = dMatrix.TryCastTo<MatrixSequence<CheckElement, CheckDevice>>();
        assert(castPtr1);
        auto castPtr2 = dMatrix2.TryCastTo<MatrixSequence<CheckElement, CheckDevice>>();
        assert(!castPtr2);
        auto castPtr3 = dMatrix.TryCastTo<Scalar<CheckElement, CheckDevice>>();
        assert(!castPtr3);
        
        auto evalRes = Evaluate(dMatrix);
        assert(evalRes == internal);

        cout << "done" << endl;
    }
    
    void test_dynamic_3d_array_sequence_case1()
    {
        cout << "Test dynamic 3d array sequence case 1...\t";
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::ThreeDArraySequence>;
        
        static_assert(IsDynamic<CheckType>);
        static_assert(IsThreeDArraySequence<CheckType>);
        
        ThreeDArraySequence<CheckElement, CheckDevice> internal(51, 7, 11, 13);
        CheckType dChecker = MakeDynamic(internal);
        
        CheckType dChecker2;
        
        assert(dChecker == dChecker);
        assert(dChecker != dChecker2);
        
        assert(!dChecker.IsEmpty());
        assert(dChecker2.IsEmpty());
        
        assert(dChecker.Shape().Length() == 51);
        assert(dChecker.Shape().PageNum() == 7);
        assert(dChecker.Shape().RowNum() == 11);
        assert(dChecker.Shape().ColNum() == 13);
        
        auto castPtr1 = dChecker.TryCastTo<ThreeDArraySequence<CheckElement, CheckDevice>>();
        assert(castPtr1);
        auto castPtr2 = dChecker2.TryCastTo<ThreeDArraySequence<CheckElement, CheckDevice>>();
        assert(!castPtr2);
        auto castPtr3 = dChecker.TryCastTo<Scalar<CheckElement, CheckDevice>>();
        assert(!castPtr3);
        
        auto evalRes = Evaluate(dChecker);
        assert(evalRes == internal);

        cout << "done" << endl;
    }
    
    void test_dynamic_batch_scalar_sequence_case1()
    {
        cout << "Test dynamic batch scalar sequence case 1...\t";
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::BatchScalarSequence>;
        
        static_assert(IsDynamic<CheckType>);
        static_assert(IsBatchScalarSequence<CheckType>);
        
        std::vector lens{3, 5, 7, 11};
        BatchScalarSequence<CheckElement, CheckDevice> internal(lens);

        CheckType dScalar = MakeDynamic(internal);
        
        CheckType dScalar2;
        
        assert(dScalar == dScalar);
        assert(dScalar != dScalar2);
        
        assert(!dScalar.IsEmpty());
        assert(dScalar2.IsEmpty());
        
        assert(dScalar.Shape().SeqLenContainer().size() == lens.size());
        assert(std::equal(dScalar.Shape().SeqLenContainer().begin(), dScalar.Shape().SeqLenContainer().end(),
                          lens.begin()));
        
        auto castPtr1 = dScalar.TryCastTo<BatchScalarSequence<CheckElement, CheckDevice>>();
        assert(castPtr1);
        auto castPtr2 = dScalar2.TryCastTo<BatchScalarSequence<CheckElement, CheckDevice>>();
        assert(!castPtr2);
        
        auto evalRes = Evaluate(dScalar);
        assert(evalRes == internal);
        cout << "done" << endl;
    }

    void test_dynamic_batch_matrix_sequence_case1()
    {
        cout << "Test dynamic batch matrix sequence case 1...\t";
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::BatchMatrixSequence>;
        
        static_assert(IsDynamic<CheckType>);
        static_assert(IsBatchMatrixSequence<CheckType>);
        
        std::vector lens{3, 5, 7, 11};
        BatchMatrixSequence<CheckElement, CheckDevice> internal(lens, 11, 13);
        CheckType dMatrix = MakeDynamic(internal);
        
        CheckType dMatrix2;
        
        assert(dMatrix == dMatrix);
        assert(dMatrix != dMatrix2);
        
        assert(!dMatrix.IsEmpty());
        assert(dMatrix2.IsEmpty());
        assert(dMatrix.Shape().SeqLenContainer().size() == lens.size());
        assert(std::equal(dMatrix.Shape().SeqLenContainer().begin(), dMatrix.Shape().SeqLenContainer().end(),
                          lens.begin()));
        assert(dMatrix.Shape().RowNum() == 11);
        assert(dMatrix.Shape().ColNum() == 13);
        
        auto castPtr1 = dMatrix.TryCastTo<BatchMatrixSequence<CheckElement, CheckDevice>>();
        assert(castPtr1);
        auto castPtr2 = dMatrix2.TryCastTo<BatchMatrixSequence<CheckElement, CheckDevice>>();
        assert(!castPtr2);
        auto castPtr3 = dMatrix.TryCastTo<Scalar<CheckElement, CheckDevice>>();
        assert(!castPtr3);
        
        auto evalRes = Evaluate(dMatrix);
        assert(evalRes == internal);

        cout << "done" << endl;
    }

    void test_dynamic_batch_3d_array_sequence_case1()
    {
        cout << "Test dynamic batch 3d array sequence case 1...\t";
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::BatchThreeDArraySequence>;
        
        static_assert(IsDynamic<CheckType>);
        static_assert(IsBatchThreeDArraySequence<CheckType>);
        
        std::vector lens{3, 5, 7, 11};
        BatchThreeDArraySequence<CheckElement, CheckDevice> internal(lens, 7, 11, 13);
        CheckType dChecker = MakeDynamic(internal);
        
        CheckType dChecker2;
        
        assert(dChecker == dChecker);
        assert(dChecker != dChecker2);
        
        assert(!dChecker.IsEmpty());
        assert(dChecker2.IsEmpty());
        
        assert(dChecker.Shape().SeqLenContainer().size() == lens.size());
        assert(std::equal(dChecker.Shape().SeqLenContainer().begin(), dChecker.Shape().SeqLenContainer().end(),
                          lens.begin()));
        assert(dChecker.Shape().PageNum() == 7);
        assert(dChecker.Shape().RowNum() == 11);
        assert(dChecker.Shape().ColNum() == 13);
        
        auto castPtr1 = dChecker.TryCastTo<BatchThreeDArraySequence<CheckElement, CheckDevice>>();
        assert(castPtr1);
        auto castPtr2 = dChecker2.TryCastTo<BatchThreeDArraySequence<CheckElement, CheckDevice>>();
        assert(!castPtr2);
        auto castPtr3 = dChecker.TryCastTo<Scalar<CheckElement, CheckDevice>>();
        assert(!castPtr3);
        
        auto evalRes = Evaluate(dChecker);
        assert(evalRes == internal);

        cout << "done" << endl;
    }
}

namespace Test::Data::General
{
    void test_dynamic()
    {
        test_dynamic_scalar_case1();
        test_dynamic_matrix_case1();
        test_dynamic_3d_array_case1();
        
        test_dynamic_batch_scalar_case1();
        test_dynamic_batch_matrix_case1();
        test_dynamic_batch_3d_array_case1();
        
        test_dynamic_scalar_sequence_case1();
        test_dynamic_matrix_sequence_case1();
        test_dynamic_3d_array_sequence_case1();
        
        test_dynamic_batch_scalar_sequence_case1();
        test_dynamic_batch_matrix_sequence_case1();
        test_dynamic_batch_3d_array_sequence_case1();
    }
}