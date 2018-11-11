#include <data/cardinal/matrix/test_trival_matrix.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_trival_matrix_case1()
    {
        cout << "Test trival matrix case 1...\t";
        static_assert(IsMatrix<TrivalMatrix<CheckElement, CheckDevice, Scalar<int, DeviceTags::CPU>>>, "Test Error");
        static_assert(IsMatrix<TrivalMatrix<CheckElement, CheckDevice, Scalar<int, DeviceTags::CPU>> &>, "Test Error");
        static_assert(IsMatrix<TrivalMatrix<CheckElement, CheckDevice, Scalar<int, DeviceTags::CPU>> &&>, "Test Error");
        static_assert(IsMatrix<const TrivalMatrix<CheckElement, CheckDevice, Scalar<int, DeviceTags::CPU>> &>, "Test Error");
        static_assert(IsMatrix<const TrivalMatrix<CheckElement, CheckDevice, Scalar<int, DeviceTags::CPU>> &&>, "Test Error");

        auto rm = MakeTrivalMatrix<CheckElement, CheckDevice>(100, 10, 20);
        assert(rm.Shape().RowNum() == 10);
        assert(rm.Shape().ColNum() == 20);

        const auto& evalHandle = rm.EvalRegister();
        EvalPlan<CheckDevice>::Eval();

        auto rm1 = evalHandle.Data();
        for (size_t i=0; i<10; ++i)
        {
            for (size_t j=0; j<20; ++j)
            {
                assert(fabs(rm1(i, j) - 100) < 0.001);
            }
        }
        cout << "done" << endl;
    }
    
    void test_trival_matrix_case2()
    {
        cout << "Test trival matrix case 2...\t";
        auto rm1 = MakeTrivalMatrix<int, CheckDevice>(14, 100, 10);
        auto rm2 = MakeTrivalMatrix<int, CheckDevice>(35, 10, 20);

        const auto& evalRes1 = rm1.EvalRegister();
        const auto& evalRes2 = rm2.EvalRegister();

        EvalPlan<CheckDevice>::Eval();
        for (size_t j = 0; j < 100; ++j)
        {
            for (size_t k = 0; k < 10; ++k)
            {
                assert(evalRes1.Data()(j, k) == 14);
            }
        }

        for (size_t j = 0; j < 10; ++j)
        {
            for (size_t k = 0; k < 20; ++k)
            {
                assert(evalRes2.Data()(j, k) == 35);
            }
        }

        cout << "done" << endl;
    }
}

namespace Test::Data::Cardinal::Matrix
{
    void test_trival_matrix()
    {
        test_trival_matrix_case1();
        test_trival_matrix_case2();
    }
}