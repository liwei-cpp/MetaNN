#include <data/test_dynamic.h>
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
        internal.Value() = 3;
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
        
        auto internal = Matrix<CheckElement, CheckDevice>::CreateWithShape(11, 13);
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
        
        auto internal = ThreeDArray<CheckElement, CheckDevice>::CreateWithShape(7, 11, 13);
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
}

namespace Test::Data
{
    void test_dynamic()
    {
        test_dynamic_scalar_case1();
        test_dynamic_matrix_case1();
        test_dynamic_3d_array_case1();
    }
}