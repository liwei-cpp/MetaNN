#include "test_eval_plan.h"
#include "../facilities/calculate_tags.h"
#include <iostream>
#include <cassert>
#include <set>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
void TestEvalPlan1()
{
    cout << "Test eval plan case 1...\t";
    auto rm = OneHotVector<int, CheckDevice>(100, 37);
    
    auto eh = rm.EvalRegister();
    try
    {
        eh.Data();
        assert(false);
    }
    catch (std::runtime_error&)
    {
    }
    cout << "done" << endl;
}

void TestEvalPlan2()
{
    cout << "Test eval plan case 2...\t";
    auto rm = OneHotVector<int, CheckDevice>(100, 37);
    
    auto eh1 = rm.EvalRegister();
    EvalPlan<CheckDevice>::Eval();
    auto eh2 = rm.EvalRegister();
    assert(eh1.Data() == eh2.Data());
    cout << "done" << endl;
}

void TestEvalPlan3()
{
    cout << "Test eval plan case 3...\t";
    auto rm = OneHotVector<int, CheckDevice>(100, 37);
    auto rm2 = rm;
    
    auto eh1 = rm.EvalRegister();
    EvalPlan<CheckDevice>::Eval();
    auto eh2 = rm2.EvalRegister();
    assert(eh1.Data() == eh2.Data());
    cout << "done" << endl;
}

void TestEvalPlan4()
{
    cout << "Test eval plan case 4...\t";
    auto rm = OneHotVector<int, CheckDevice>(100, 37);
    
    auto eh1 = rm.EvalRegister();
    auto eh2 = rm.EvalRegister();
    EvalPlan<CheckDevice>::Eval();
    assert(eh1.Data() == eh2.Data());
    cout << "done" << endl;
}
}

void test_eval_plan()
{
    TestEvalPlan1();
    TestEvalPlan2();
    TestEvalPlan3();
    TestEvalPlan4();
}