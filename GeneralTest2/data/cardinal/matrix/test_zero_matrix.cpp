#include <data/cardinal/matrix/test_zero_matrix.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_zero_matrix_case1()
    {
        cout << "Test zero matrix case 1...\t";
        static_assert(IsMatrix<ZeroMatrix<CheckElement, CheckDevice>>);
        static_assert(IsMatrix<ZeroMatrix<CheckElement, CheckDevice> &>);
        static_assert(IsMatrix<ZeroMatrix<CheckElement, CheckDevice> &&>);
        static_assert(IsMatrix<const ZeroMatrix<CheckElement, CheckDevice> &>);
        static_assert(IsMatrix<const ZeroMatrix<CheckElement, CheckDevice> &&>);

        auto rm = ZeroMatrix<CheckElement, CheckDevice>::CreateWithShape(10, 20);
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
}

namespace Test::Data::Cardinal::Matrix
{
    void test_zero_matrix()
    {
        test_zero_matrix_case1();
    }
}