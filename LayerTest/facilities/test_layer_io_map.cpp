#include "test_layer_io_map.h"
#include <MetaNN/meta_nn2.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_layer_io_map1()
    {
        cout << "Test layer io map case 1...\t";
        
        struct Key1; struct Key2; struct Key3;
        struct Value1; struct Value2; struct Value3;
        using CheckType = LayerIOMap<LayerKV<Key1, Value1>, LayerKV<Key2, Value2>, LayerKV<Key3, Value3>>;
        
        static_assert(std::is_same_v<CheckType::Find<Key1>, Value1>);
        static_assert(std::is_same_v<CheckType::Find<Key2>, Value2>);
        static_assert(std::is_same_v<CheckType::Find<Key3>, Value3>);
        cout << "done" << endl;
    }
}

namespace Test::Layer
{
    void test_layer_io_map()
    {
        test_layer_io_map1();
    }
}