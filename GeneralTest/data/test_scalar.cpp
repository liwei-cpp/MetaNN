#include "test_scalar.h"
#include <iostream>
#include <cassert>
#include <set>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
void test_scalar1()
{
    cout << "Test scalar case 1...\t";
    static_assert(IsScalar<Scalar<int, DeviceTags::CPU>>, "Test Error");
    static_assert(IsScalar<Scalar<int, DeviceTags::CPU> &>, "Test Error");
    static_assert(IsScalar<Scalar<int, DeviceTags::CPU> &&>, "Test Error");
    static_assert(IsScalar<const Scalar<int, DeviceTags::CPU> &>, "Test Error");
    static_assert(IsScalar<const Scalar<int, DeviceTags::CPU> &&>, "Test Error");
    
    Scalar<float, DeviceTags::CPU> pi(3.1415926f);
    assert(pi == pi);
    
    auto x = pi.EvalRegister();
    assert(x.Data() == pi);
    cout << "done" << endl;
}    
}

void test_scalar()
{
    test_scalar1();
}