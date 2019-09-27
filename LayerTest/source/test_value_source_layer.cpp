#include <MetaNN/meta_nn2.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_value_source_layer1()
    {
        cout << "Test value source layer case 1...\t";
        using RootLayer = MakeInferLayer<ValueSourceLayer>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");
        
        RootLayer layer("root");
        auto out = layer.FeedForward(NullParameter{});
        
        auto res = out.Get<LayerOutput>();
        assert(fabs(res < 0.001));

        cout << "done" << endl;
    }
    
    void test_value_source_layer2()
    {
        cout << "Test value source layer case 2...\t";
        using RootLayer = MakeInferLayer<ValueSourceLayer, PNumeratorIs<-1>, PDenominatorIs<2>>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");
        
        RootLayer layer("root");
        auto out = layer.FeedForward(NullParameter{});
        
        auto res = out.Get<LayerOutput>();
        assert(fabs(res + 0.5 < 0.001));

        cout << "done" << endl;
    }
    
    void test_value_source_layer3()
    {
        cout << "Test value source layer case 3...\t";
        using RootLayer = MakeInferLayer<ValueSourceLayer, PValueTypeIs<int>, PNumeratorIs<3>, PDenominatorIs<2>>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");
        
        RootLayer layer("root");
        auto out = layer.FeedForward(NullParameter{});
        
        auto res = out.Get<LayerOutput>();
        static_assert(std::is_same_v<decltype(res), int>);
        assert(fabs(res - 1 < 0.001));

        cout << "done" << endl;
    }
}

namespace Test::Layer::Source
{
    void test_value_source_layer()
    {
        test_value_source_layer1();
        test_value_source_layer2();
        test_value_source_layer3();
    }
}