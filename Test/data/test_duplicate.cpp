#include "test_duplicate.h"
#include "../facilities/calculate_tags.h"
#include <iostream>
#include <cassert>
#include <set>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
void TestDuplicate1()
{
    cout << "Test duplicate case 1 (matrix)...\t";
    static_assert(IsBatchMatrix<Duplicate<Matrix<CheckElement, CheckDevice>>>, "Test Error");
    static_assert(IsBatchMatrix<Duplicate<Matrix<CheckElement, CheckDevice>> &>, "Test Error");
    static_assert(IsBatchMatrix<Duplicate<Matrix<CheckElement, CheckDevice>> &&>, "Test Error");
    static_assert(IsBatchMatrix<const Duplicate<Matrix<CheckElement, CheckDevice>> &>, "Test Error");
    static_assert(IsBatchMatrix<const Duplicate<Matrix<CheckElement, CheckDevice>> &&>, "Test Error");

    auto me1 = Matrix<CheckElement, CheckDevice>(10, 20);
    int c = 0;
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
        {
            me1.SetValue(i, j, (float)(c++));
        }
    }
    auto rm1 = MakeDuplicate(13, me1);
    assert(rm1.BatchNum() == 13);
    assert(rm1.RowNum() == 10);
    assert(rm1.ColNum() == 20);
    
    auto rm2 = Evaluate(rm1);
    for (size_t i = 0; i < 13; ++i)
    {
        for (size_t j = 0; j < 10; ++j)
        {
            for (size_t k = 0; k < 20; ++k)
            {
                assert(rm2[i](j, k) == me1(j, k));
            }
        }
    }
    cout << "done" << endl;
}

void TestDuplicate2()
{
    cout << "Test duplicate case 2 (scalar)...\t";
    static_assert(IsBatchScalar<Duplicate<Scalar<CheckElement, CheckDevice>>>, "Test Error");
    static_assert(IsBatchScalar<Duplicate<Scalar<CheckElement, CheckDevice>> &>, "Test Error");
    static_assert(IsBatchScalar<Duplicate<Scalar<CheckElement, CheckDevice>> &&>, "Test Error");
    static_assert(IsBatchScalar<const Duplicate<Scalar<CheckElement, CheckDevice>> &>, "Test Error");
    static_assert(IsBatchScalar<const Duplicate<Scalar<CheckElement, CheckDevice>> &&>, "Test Error");

    auto rm1 = Duplicate<Scalar<CheckElement, CheckDevice>>(3, 13);
    assert(rm1.Size() == 13);

    auto evalHandle = rm1.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();
    auto rm2 = evalHandle.Data();

    for (size_t i = 0; i < 13; ++i)
    {
        assert(rm2[i] == 3);
    }
    cout << "done" << endl;
}
}

void test_duplicate()
{
    TestDuplicate1();
    TestDuplicate2();
}