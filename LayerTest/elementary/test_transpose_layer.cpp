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
    void test_transpose_layer1()
    {
        cout << "Test transpose layer case 1 ...\t";
        using RootLayer = MakeInferLayer<TransposeLayer>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        auto in = GenMatrix<CheckElement>(4, 5, -3.3f, 0.1f);
        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
    
        assert(res.Shape().RowNum() == 5);
        assert(res.Shape().ColNum() == 4);
    
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                assert(fabs(res(j, i) - in(i, j)) < 0.0001);
            }
        }

        NullParameter fbIn;
        auto out_grad = layer.FeedBackward(fbIn);
        auto fb1 = out_grad.Get<LayerInput>();
        static_assert(std::is_same<decltype(fb1), NullParameter>::value, "Test error");

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
    
    void test_transpose_layer2()
    {
        cout << "Test transpose layer case 2 ...\t";
        using RootLayer = MakeTrainLayer<TransposeLayer, CommonInputMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        auto in = GenMatrix<CheckElement>(4, 5, -3.3f, 0.1f);
        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(res.Shape().RowNum() == 5);
        assert(res.Shape().ColNum() == 4);
    
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                assert(fabs(res(j, i) - in(i, j)) < 0.0001);
            }
        }

        auto grad = GenMatrix<float>(5, 4, 1.8f, -0.2f);
        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));
        auto fb = Evaluate(out_grad.Get<LayerInput>());

        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                assert(fabs(fb(i, j) - grad(j, i)) < 0.0001);
            }
        }

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
}

namespace Test::Layer::Elementary
{
    void test_transpose_layer()
    {
        test_transpose_layer1();
        test_transpose_layer2();
    }
}