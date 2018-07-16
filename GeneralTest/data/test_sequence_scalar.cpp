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
    cout << "Test scalar sequence case 1...\t";
    static_assert(IsScalarSequence<Sequence<CheckElement, CheckDevice, CategoryTags::Scalar>>, "Test Error");
    static_assert(IsScalarSequence<Sequence<CheckElement, CheckDevice, CategoryTags::Scalar>&>, "Test Error");
    static_assert(IsScalarSequence<Sequence<CheckElement, CheckDevice, CategoryTags::Scalar>&&>, "Test Error");
    static_assert(IsScalarSequence<const Sequence<CheckElement, CheckDevice, CategoryTags::Scalar>&>, "Test Error");
    static_assert(IsScalarSequence<const Sequence<CheckElement, CheckDevice, CategoryTags::Scalar>&&>, "Test Error");

    Sequence<CheckElement, CheckDevice, CategoryTags::Scalar> check;
    assert(check.Length() == 0);

    check = Sequence<CheckElement, CheckDevice, CategoryTags::Scalar>(13);
    assert(check.Length() == 13);

    int c = 0;
    for (size_t i=0; i<13; ++i)
    {
        check.SetValue(i, (float)(c++));
    }

    const Sequence<CheckElement, CheckDevice, CategoryTags::Scalar> c2 = check;
    c = 0;
    for (size_t i=0; i<13; ++i)
    {
        assert(c2[i] == (float)(c++));
    }

    auto evalHandle = check.EvalRegister();
    auto cm = evalHandle.Data();

    for (size_t i = 0; i < cm.Length(); ++i)
    {
        assert(cm[i] == check[i]);
    }
    cout << "done" << endl;
}
}

void test_sequence_scalar()
{
    test_case1();
}