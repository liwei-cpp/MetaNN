#include "test_collapse.h"
#include "../facilities/data_gen.h"

#include <MetaNN/meta_nn.h>
#include <cassert>
#include <cmath>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_collapse1()
{
    cout << "Test collapse case 1 ...\t";
    auto rm1 = GenBatchMatrix<float>(4, 5, 7, 1.0f, 0.0001f);
    auto t = Collapse(rm1);
    assert(t.RowNum() == 4);
    assert(t.ColNum() == 5);
    
    auto handle = t.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();
    auto t_r = handle.Data();
    
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j<5; ++j)
        {
            float aim = 0;
            for (size_t k = 0; k < 7; ++k)
            {
                aim += rm1[k](i, j);
            }
            assert(fabs(t_r(i, j) - aim) < 0.0001);
        }
    }
    cout << "done" << endl;
}

void test_collapse2()
{
    cout << "Test collapse case 2 ...\t";
    {
        auto rm1 = GenBatchMatrix<float>(4, 5, 7, 1.0f, 0.0001f);
        auto collapse1 = Collapse(rm1);
        auto collapse2 = Collapse(rm1);

        assert(collapse1 == collapse2);

        auto handle1 = collapse1.EvalRegister();
        auto handle2 = collapse1.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();
        auto cm1 = handle1.Data();
        auto cm2 = handle2.Data();
        assert(cm1 == cm2);
    }
    {
        auto rm1 = GenBatchMatrix<float>(4, 5, 7, 1.0f, 0.0001f);
        auto collapse1 = Collapse(rm1);
        auto collapse2 = collapse1;

        assert(collapse1 == collapse2);

        const auto& evalHandle1 = collapse1.EvalRegister();
        const auto& evalHandle2 = collapse2.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto cm1 = evalHandle1.Data();
        auto cm2 = evalHandle2.Data();
        assert(cm1 == cm2);
    }
    cout << "done" << endl;
}
}

void test_collapse()
{
    test_collapse1();
    test_collapse2();
}
