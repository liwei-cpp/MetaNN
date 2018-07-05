#include "test_general_matrix.h"
#include "../facilities/calculate_tags.h"
#include <iostream>
#include <cassert>
#include <set>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
void TestArray1()
{
    cout << "Test array case 1 (matrix array)...\t";
    static_assert(IsBatchMatrix<Array<Matrix<CheckElement, CheckDevice>>>, "Test Error");
    static_assert(IsBatchMatrix<Array<Matrix<CheckElement, CheckDevice>> &>, "Test Error");
    static_assert(IsBatchMatrix<Array<Matrix<CheckElement, CheckDevice>> &&>, "Test Error");
    static_assert(IsBatchMatrix<const Array<Matrix<CheckElement, CheckDevice>> &>, "Test Error");
    static_assert(IsBatchMatrix<const Array<Matrix<CheckElement, CheckDevice>> &&>, "Test Error");

    auto rm1 = Array<Matrix<CheckElement, CheckDevice>>(10, 20);
    assert(rm1.BatchNum() == 0);
    assert(rm1.empty());

    int c = 0;
    auto me1 = Matrix<CheckElement, CheckDevice>(10, 20);
    auto me2 = Matrix<CheckElement, CheckDevice>(10, 20);
    auto me3 = Matrix<CheckElement, CheckDevice>(10, 20);
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
        {
            me1.SetValue(i, j, (float)(c++));
            me2.SetValue(i, j, (float)(c++));
            me3.SetValue(i, j, (float)(c++));
        }
    }
    rm1.push_back(me1);
    rm1.push_back(me2);
    rm1.push_back(me3);
    assert(rm1.BatchNum() == 3);
    assert(!rm1.empty());

    auto evalHandle = rm1.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();
    auto rm2 = evalHandle.Data();
    
    for (size_t i = 0; i < 10; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
        {
            assert(rm1[0](i, j) == me1(i, j));
            assert(rm1[1](i, j) == me2(i, j));
            assert(rm1[2](i, j) == me3(i, j));
        }
    }
    cout << "done" << endl;
}

void TestArray2()
{
    cout << "Test array case 2 (scalar array)...\t";
    static_assert(IsBatchScalar<Array<Scalar<CheckElement, CheckDevice>>>, "Test Error");
    static_assert(IsBatchScalar<Array<Scalar<CheckElement, CheckDevice>> &>, "Test Error");
    static_assert(IsBatchScalar<Array<Scalar<CheckElement, CheckDevice>> &&>, "Test Error");
    static_assert(IsBatchScalar<const Array<Scalar<CheckElement, CheckDevice>> &>, "Test Error");
    static_assert(IsBatchScalar<const Array<Scalar<CheckElement, CheckDevice>> &&>, "Test Error");

    auto rm1 = Array<Scalar<CheckElement, CheckDevice>>();
    assert(rm1.BatchNum() == 0);
    assert(rm1.empty());

    rm1.push_back(3);
    rm1.push_back(8);
    rm1.push_back(2);
    assert(rm1.BatchNum() == 3);
    assert(!rm1.empty());

    auto evalHandle = rm1.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();
    auto rm2 = evalHandle.Data();
    
    assert(rm2[0] == 3);
    assert(rm2[1] == 8);
    assert(rm2[2] == 2);
    cout << "done" << endl;
}

void TestArray3()
{
    cout << "Test array case 3 ...\t";

    {
        vector<Matrix<CheckElement, CheckDevice>> check;
        check.emplace_back(10, 16);
        check.emplace_back(10, 16);
        check.emplace_back(10, 16);
    
        auto tmp = MakeArray(check.begin(), check.end());
        assert(tmp.RowNum() == 10);
        assert(tmp.ColNum() == 16);
        assert(tmp.BatchNum() == 3);
    }
    {
        vector<Scalar<int, CheckDevice>> check{3, 5, 8, 6};    
        auto tmp = MakeArray(check.begin(), check.end());
        assert(tmp.BatchNum() == 4);
        assert(tmp[0].Value() == 3);
        assert(tmp[1].Value() == 5);
        assert(tmp[2].Value() == 8);
        assert(tmp[3].Value() == 6);
    }
    cout << "done" << endl;
}
}

void test_array()
{
    TestArray1();
    TestArray2();
    TestArray3();
}