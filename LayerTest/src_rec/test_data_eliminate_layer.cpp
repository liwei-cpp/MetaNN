#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <data_gen.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    using CommonInputMap = LayerIOMap<LayerKV<LayerInput, Matrix<CheckElement, CheckDevice>>>;

    void test_data_eliminate_layer1()
    {
        cout << "Test data eliminate layer case 1 ...\t";
        using RootLayer = MakeLayer<DataEliminateLayer, CommonInputMap>;
        static_assert(!RootLayer::IsFeedbackOutput);
        static_assert(!RootLayer::IsUpdate);
        
        RootLayer layer("root");
        
        auto mat = GenMatrix<CheckElement>(10, 3);
        auto res = layer.FeedForward(LayerInputCont<RootLayer>().Set<LayerInput>(mat));
        static_assert(res.Length == 0);
        cout << "done" << endl;
    }
    
    void test_data_eliminate_layer2()
    {
        cout << "Test data eliminate layer case 2 ...\t";
        using RootLayer = MakeBPLayer<DataEliminateLayer, CommonInputMap, NullParameter, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput);
        static_assert(!RootLayer::IsUpdate);
        
        RootLayer layer("root");
        
        auto mat = GenMatrix<CheckElement>(10, 3);
        auto res = layer.FeedForward(LayerInputCont<RootLayer>().Set<LayerInput>(mat));
        static_assert(res.Length == 0);
        
        auto check = layer.FeedBackward(LayerOutputCont<RootLayer>()).Get<LayerInput>();
        auto checkRes = Evaluate(check);
        assert(checkRes.Shape() == mat.Shape());
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(checkRes(i,j)) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Layer::SrcRec
{
    void test_data_eliminate_layer()
    {
        test_data_eliminate_layer1();
        test_data_eliminate_layer2();
    }
}