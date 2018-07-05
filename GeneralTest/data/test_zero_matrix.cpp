#include "test_zero_matrix.h"
#include "../facilities/calculate_tags.h"
#include <iostream>
#include <cassert>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
void TestZeroMatrix1()
{
    cout << "Test zero matrix case 1...\t";
    static_assert(IsMatrix<ZeroMatrix<int, CheckDevice>>, "Test Error");
    static_assert(IsMatrix<ZeroMatrix<int, CheckDevice> &>, "Test Error");
    static_assert(IsMatrix<ZeroMatrix<int, CheckDevice> &&>, "Test Error");
    static_assert(IsMatrix<const ZeroMatrix<int, CheckDevice> &>, "Test Error");
    static_assert(IsMatrix<const ZeroMatrix<int, CheckDevice> &&>, "Test Error");

    auto rm = ZeroMatrix<int, CheckDevice>(10, 20);
    assert(rm.RowNum() == 10);
    assert(rm.ColNum() == 20);

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

void test_zero_matrix()
{
    TestZeroMatrix1();
}
