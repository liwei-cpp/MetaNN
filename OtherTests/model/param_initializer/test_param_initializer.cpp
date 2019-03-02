#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_param_initializer1()
    {
        struct Key1; struct Key2;
        cout << "test param initializer case 1 ...";
        auto check
            = MakeInitializer<CheckElement>(InitializerKV<Key1>(3),
                                            InitializerKV<Key2>(1.5));
        assert(check.GetFiller<Key1>() == 3);
        assert(fabs(check.GetFiller<Key2>() - 1.5) < 0.001);
        cout << "done" << endl;
    }
}

namespace Test::Model::ParamInitializer
{
    void test_param_initializer()
    {
        test_param_initializer1();
    }
}