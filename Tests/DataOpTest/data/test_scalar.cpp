#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_scalar_case1()
    {
        cout << "Test scalar case 1...\t";
        static_assert(IsScalar<Scalar<CheckElement, CheckDevice>>, "Test Error");
        static_assert(IsScalar<Scalar<CheckElement, CheckDevice> &>, "Test Error");
        static_assert(IsScalar<Scalar<CheckElement, CheckDevice> &&>, "Test Error");
        static_assert(IsScalar<const Scalar<CheckElement, CheckDevice> &>, "Test Error");
        static_assert(IsScalar<const Scalar<CheckElement, CheckDevice> &&>, "Test Error");
    
        Scalar<CheckElement, CheckDevice> pi(3.1415926f);
        Scalar<CheckElement, CheckDevice> value2(3.14);
        assert(pi == pi);
        assert(!(pi != pi));
        assert(pi != value2);
    
        auto x = pi.EvalRegister();
        assert(x.Data() == pi);
        cout << "done" << endl;
    }
}

namespace Test::Data
{
    void test_scalar()
    {
        test_scalar_case1();
    }
}