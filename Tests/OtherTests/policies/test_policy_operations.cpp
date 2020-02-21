#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
    struct Tag1;
    struct Tag2;
    struct Tag3;

    void test_policy_operations1()
    {
        cout << "Test policy operations case 1...\t";
        using input = PolicyContainer<PUpdate,
                                      SubPolicyContainer<Tag1, PNoUpdate,
                                                         SubPolicyContainer<Tag2>>>;
        using check1 = SubPolicyPicker<input, Tag3>;
        static_assert(is_same_v<check1, PolicyContainer<PUpdate>>);

        using check2 = SubPolicyPicker<input, Tag1>;
        static_assert(is_same_v<check2, PolicyContainer<PNoUpdate, SubPolicyContainer<Tag2>>>);

        using check3 = SubPolicyPicker<check2, Tag3>;
        static_assert(is_same_v<check3, PolicyContainer<PNoUpdate>>);

        using check4 = SubPolicyPicker<check2, Tag2>;
        static_assert(is_same_v<check4, PolicyContainer<PNoUpdate>>);
        cout << "done" << endl;
    }
}

namespace Test::Policies
{
    void test_policy_operations()
    {
        test_policy_operations1();
    }
}
