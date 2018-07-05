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
void test_case1()
{
    cout << "Test batch scalar case 1...\t";
    static_assert(IsBatchScalar<Batch<CheckElement, CheckDevice, CategoryTags::Scalar>>, "Test Error");
    static_assert(IsBatchScalar<Batch<CheckElement, CheckDevice, CategoryTags::Scalar>&>, "Test Error");
    static_assert(IsBatchScalar<Batch<CheckElement, CheckDevice, CategoryTags::Scalar>&&>, "Test Error");
    static_assert(IsBatchScalar<const Batch<CheckElement, CheckDevice, CategoryTags::Scalar>&>, "Test Error");
    static_assert(IsBatchScalar<const Batch<CheckElement, CheckDevice, CategoryTags::Scalar>&&>, "Test Error");

    Batch<CheckElement, CheckDevice, CategoryTags::Scalar> check;
    assert(check.BatchNum() == 0);

    check = Batch<CheckElement, CheckDevice, CategoryTags::Scalar>(13);
    assert(check.BatchNum() == 13);

    int c = 0;
    for (size_t i=0; i<13; ++i)
    {
        check.SetValue(i, (float)(c++));
    }

    const Batch<CheckElement, CheckDevice, CategoryTags::Scalar> c2 = check;
    c = 0;
    for (size_t i=0; i<13; ++i)
    {
        assert(c2[i] == (float)(c++));
    }

    auto evalHandle = check.EvalRegister();
    auto cm = evalHandle.Data();

    for (size_t i = 0; i < cm.BatchNum(); ++i)
    {
        assert(cm[i] == check[i]);
    }
    cout << "done" << endl;
}
}

void test_batch_scalar()
{
    test_case1();
}