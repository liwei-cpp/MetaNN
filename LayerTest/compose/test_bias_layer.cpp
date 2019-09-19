#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <data_gen.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
    using CommonInputMap = LayerIOMap<LayerKV<LayerInput, Matrix<CheckElement, CheckDevice>>>;
    using CommonGradMap = LayerIOMap<LayerKV<LayerOutput, Matrix<CheckElement, CheckDevice>>>;

    void test_bias_layer1()
    {
        cout << "Test bias layer case 1 ...\t";
/*        using RootLayer = MakeLayer<BiasLayer, CommonInputMap>;
        static_assert(!RootLayer::IsUpdate, "Test Error");
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");

        RootLayer layer("root", 2, 1);

        // Initialization
        auto filler = MakeInitializer<CheckElement>();
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;

        auto mat = GenMatrix<CheckElement>(2, 1);
        filler.SetParam("root-param", mat);
        
        layer.Init(filler, loadBuffer);


        auto input = GenMatrix<CheckElement>(2, 1, 0.5f, -0.1f);
        auto bi = LayerInputCont<RootLayer>().Set<LayerInput>(input);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(bi);
        auto res = Evaluate(out.Get<LayerIO>());
        assert(fabs(res(0, 0) - input(0, 0) - weight(0, 0)) < 0.001);
        assert(fabs(res(1, 0) - input(1, 0) - weight(1, 0)) < 0.001);

        auto fbIn = LayerIO::Create();
        auto out_grad = layer.FeedBackward(fbIn);
        auto fbOut = out_grad.Get<LayerIO>();
        static_assert(is_same<decltype(fbOut), NullParameter>::value, "Test error");

        params.clear();
        layer.SaveWeights(params);
        assert(params.find("root") != params.end());

        LayerNeutralInvariant(layer);*/
        cout << "done" << endl;
    }
}
namespace Test::Layer::Compose
{
    void test_bias_layer()
    {
        test_bias_layer1();
    }
}