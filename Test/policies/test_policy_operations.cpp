#include <iostream>
#include <cassert>
#include <set>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
struct Tag1;
struct Tag2;
struct Tag3;

void test_policy_operations1()
{
    cout << "Test policy operations case 1...\t";
    using input = PolicyContainer<PBatchMode,
                                  SubPolicyContainer<Tag1, PNoBatchMode,
                                                     SubPolicyContainer<Tag2>>>;
    using check1 = SubPolicyPicker<input, Tag3>;
    static_assert(is_same<check1, PolicyContainer<PBatchMode>>::value, "Check Error");

    using check2 = SubPolicyPicker<input, Tag1>;
    static_assert(is_same<check2, PolicyContainer<PNoBatchMode, SubPolicyContainer<Tag2>>>::value, "Check Error");

    using check3 = SubPolicyPicker<check2, Tag3>;
    static_assert(is_same<check3, PolicyContainer<PNoBatchMode>>::value, "Check Error");

    using check4 = SubPolicyPicker<check2, Tag2>;
    static_assert(is_same<check4, PolicyContainer<PNoBatchMode>>::value, "Check Error");
    cout << "done" << endl;
}
}
void test_policy_operations()
{
    test_policy_operations1();
}
