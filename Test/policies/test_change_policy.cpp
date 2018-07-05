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

void test_change_policy1()
{
    cout << "Test change policy case 1...\t";
    using input = PolicyContainer<PBatchMode,
                                  SubPolicyContainer<Tag1, PNoBatchMode>>;

    using check1 = ChangePolicy<PEnableBptt, input>;
    static_assert(is_same<check1, PolicyContainer<PBatchMode, SubPolicyContainer<Tag1, PNoBatchMode>, PEnableBptt>>::value, "Check Error");

    using check2 = ChangePolicy<PNoBatchMode, input>;
    static_assert(is_same<check2, PolicyContainer<SubPolicyContainer<Tag1, PNoBatchMode>, PNoBatchMode>>::value, "Check Error");
    cout << "done" << endl;
}
}
void test_change_policy()
{
    test_change_policy1();
}
