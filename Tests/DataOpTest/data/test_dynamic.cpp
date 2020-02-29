#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_dynamic_scalar_case1()
    {
        cout << "Test dynamic scalar case 1...\t";
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::Tensor<0>>;
        
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
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::Tensor<2>>;
        
        static_assert(IsDynamic<CheckType>);
        static_assert(IsMatrix<CheckType>);
        
        Matrix<CheckElement, CheckDevice> internal(11, 13);
        CheckType dMatrix = MakeDynamic(internal);
        
        CheckType dMatrix2;
        
        assert(dMatrix == dMatrix);
        assert(dMatrix != dMatrix2);
        
        assert(!dMatrix.IsEmpty());
        assert(dMatrix2.IsEmpty());
        
        assert(dMatrix.Shape()[0] == 11);
        assert(dMatrix.Shape()[1] == 13);
        
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
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::Tensor<3>>;
        
        static_assert(IsDynamic<CheckType>);
        static_assert(IsThreeDArray<CheckType>);
        
        ThreeDArray<CheckElement, CheckDevice> internal(7, 11, 13);
        CheckType dChecker = MakeDynamic(internal);
        
        CheckType dChecker2;
        
        assert(dChecker == dChecker);
        assert(dChecker != dChecker2);
        
        assert(!dChecker.IsEmpty());
        assert(dChecker2.IsEmpty());
        
        assert(dChecker.Shape()[0] == 7);
        assert(dChecker.Shape()[1] == 11);
        assert(dChecker.Shape()[2] == 13);
        
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
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::Tensor<1>>;
        
        static_assert(IsDynamic<CheckType>);
        static_assert(IsTensorWithDim<CheckType, 1>);
        
        Tensor<CheckElement, CheckDevice, 1> internal{11};
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
        
        assert(dScalar.Shape()[0] == 11);
        
        auto castPtr1 = dScalar.TryCastTo<Tensor<CheckElement, CheckDevice, 1>>();
        assert(castPtr1);
        auto castPtr2 = dScalar2.TryCastTo<Tensor<CheckElement, CheckDevice, 1>>();
        assert(!castPtr2);
        
        auto evalRes = Evaluate(dScalar);
        assert(evalRes == internal);
        cout << "done" << endl;
    }

    void test_dynamic_batch_3d_array_case1()
    {
        cout << "Test dynamic batch 3d array case 1...\t";
        using CheckType = DynamicData<CheckElement, CheckDevice, CategoryTags::Tensor<4>>;
        
        static_assert(IsDynamic<CheckType>);
        static_assert(IsTensorWithDim<CheckType, 4>);
        
        Tensor<CheckElement, CheckDevice, 4> internal(51, 7, 11, 13);
        CheckType dChecker = MakeDynamic(internal);
        
        CheckType dChecker2;
        
        assert(dChecker == dChecker);
        assert(dChecker != dChecker2);
        
        assert(!dChecker.IsEmpty());
        assert(dChecker2.IsEmpty());
        
        assert(dChecker.Shape()[0] == 51);
        assert(dChecker.Shape()[1] == 7);
        assert(dChecker.Shape()[2] == 11);
        assert(dChecker.Shape()[3] == 13);
        
        auto castPtr1 = dChecker.TryCastTo<Tensor<CheckElement, CheckDevice, 4>>();
        assert(castPtr1);
        auto castPtr2 = dChecker2.TryCastTo<Tensor<CheckElement, CheckDevice, 4>>();
        assert(!castPtr2);
        auto castPtr3 = dChecker.TryCastTo<Scalar<CheckElement, CheckDevice>>();
        assert(!castPtr3);
        
        auto evalRes = Evaluate(dChecker);
        assert(evalRes == internal);

        cout << "done" << endl;
    }
}

namespace Test::Data
{
    void test_dynamic()
    {
        test_dynamic_scalar_case1();
        test_dynamic_matrix_case1();
        test_dynamic_3d_array_case1();
        
        test_dynamic_batch_scalar_case1();
        test_dynamic_batch_3d_array_case1();
    }
}