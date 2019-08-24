#include <MetaNN/meta_nn2.h>
#include <data_gen.h>
#include <calculate_tags.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
    using CommonInputMap = LayerIOMap<LayerKV<LayerInput, Matrix<CheckElement, CheckDevice>>>;
    using CommonGradMap = LayerIOMap<LayerKV<LayerOutput, Matrix<CheckElement, CheckDevice>>>;
    
    void test_relu_layer1()
    {
        cout << "Test ReLU layer case 1 ...\t";
        using RootLayer = MakeLayer<ReLULayer, CommonInputMap>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        Matrix<CheckElement, CheckDevice> in(3, 1);
        in.SetValue(0, 0, -0.27f);
        in.SetValue(1, 0, 0.41f);
        in.SetValue(2, 0, 0.0f);

        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(fabs(res(0, 0)) < 0.001);
        assert(fabs(res(1, 0) - 0.41f) < 0.001);
        assert(fabs(res(2, 0)) < 0.001);

        NullParameter fbIn;
        auto out_grad = layer.FeedBackward(fbIn);
        auto fb1 = out_grad.Get<LayerInput>();
        static_assert(std::is_same<decltype(fb1), NullParameter>::value, "Test error");

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }

    void test_relu_layer2()
    {
        cout << "Test ReLU layer case 2 ...\t";
        using RootLayer = MakeBPLayer<ReLULayer, CommonInputMap, CommonGradMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        Matrix<CheckElement, CheckDevice> in(3, 1);
        in.SetValue(0, 0, -0.27f);
        in.SetValue(1, 0, 0.41f);

        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(fabs(res(0, 0)) < 0.001);
        assert(fabs(res(1, 0) - 0.41f) < 0.001);
        assert(fabs(res(2, 0)) < 0.001);

        Matrix<CheckElement, CheckDevice> grad(3, 1);
        grad.SetValue(0, 0, 0.1f);
        grad.SetValue(1, 0, 0.3f);
        grad.SetValue(2, 0, 0.4f);

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));
        auto fb = Evaluate(out_grad.Get<LayerInput>());
        assert(fabs(fb(0, 0)) < 0.001);
        assert(fabs(fb(1, 0) - 0.3f) < 0.001);
        assert(fabs(fb(2, 0)) < 0.001);

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
}

namespace Test::Layer::Elementary
{
    void test_relu_layer()
    {
        test_relu_layer1();
        test_relu_layer2();
    }
}
