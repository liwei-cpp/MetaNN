#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_zero_data1()
    {
        cout << "Test zero data case 1 (scalar) ...\t";
        static_assert(IsScalar<ZeroData<CategoryTags::Scalar, CheckElement, CheckDevice>>);
        static_assert(IsScalar<ZeroData<CategoryTags::Scalar, CheckElement, CheckDevice> &>);
        static_assert(IsScalar<ZeroData<CategoryTags::Scalar, CheckElement, CheckDevice> &&>);
        static_assert(IsScalar<const ZeroData<CategoryTags::Scalar, CheckElement, CheckDevice> &>);
        static_assert(IsScalar<const ZeroData<CategoryTags::Scalar, CheckElement, CheckDevice> &&>);
    
        ZeroData<CategoryTags::Scalar, CheckElement, CheckDevice> val1;
        assert(val1 == val1);
    
        auto x = Evaluate(val1);
        assert(x.Value() == 0);
        cout << "done" << endl;
    }
    
    void test_zero_data2()
    {
        cout << "Test zero data case 2 (matrix) ...\t";
        static_assert(IsMatrix<ZeroData<CategoryTags::Matrix, CheckElement, CheckDevice>>);
        static_assert(IsMatrix<ZeroData<CategoryTags::Matrix, CheckElement, CheckDevice> &>);
        static_assert(IsMatrix<ZeroData<CategoryTags::Matrix, CheckElement, CheckDevice> &&>);
        static_assert(IsMatrix<const ZeroData<CategoryTags::Matrix, CheckElement, CheckDevice> &>);
        static_assert(IsMatrix<const ZeroData<CategoryTags::Matrix, CheckElement, CheckDevice> &&>);

        ZeroData<CategoryTags::Matrix, CheckElement, CheckDevice> rm(10, 20);
        assert(rm.Shape().RowNum() == 10);
        assert(rm.Shape().ColNum() == 20);

        const auto& evalHandle = rm.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto rm1 = evalHandle.Data();
        for (size_t i=0; i<10; ++i)
        {
            for (size_t j=0; j<20; ++j)
            {
                assert(rm1(i, j) == 0);
            }
        }

        cout << "done" << endl;
    }
    
    void test_zero_data3()
    {
        cout << "Test zero data case 3 (3d array) ...\t";
        static_assert(IsThreeDArray<ZeroData<CategoryTags::ThreeDArray, CheckElement, CheckDevice>>);
        static_assert(IsThreeDArray<ZeroData<CategoryTags::ThreeDArray, CheckElement, CheckDevice>&>);
        static_assert(IsThreeDArray<ZeroData<CategoryTags::ThreeDArray, CheckElement, CheckDevice>&&>);
        static_assert(IsThreeDArray<const ZeroData<CategoryTags::ThreeDArray, CheckElement, CheckDevice>&>);
        static_assert(IsThreeDArray<const ZeroData<CategoryTags::ThreeDArray, CheckElement, CheckDevice>&&>);

        ZeroData<CategoryTags::ThreeDArray, CheckElement, CheckDevice> rm;
        assert(rm.Shape().PageNum() == 0);
        assert(rm.Shape().RowNum() == 0);
        assert(rm.Shape().ColNum() == 0);

        rm = ZeroData<CategoryTags::ThreeDArray, CheckElement, CheckDevice>(5, 10, 20);
        assert(rm.Shape().PageNum() == 5);
        assert(rm.Shape().RowNum() == 10);
        assert(rm.Shape().ColNum() == 20);

        auto res = Evaluate(rm);
        for (size_t p = 0; p < 5; ++p)
        {
            for (size_t i=0; i<10; ++i)
            {
                for (size_t j=0; j<20; ++j)
                {
                    assert(res(p, i, j) == 0);
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Data::General
{
    void test_zero_data()
    {
        test_zero_data1();
        test_zero_data2();
        test_zero_data3();
    }
}