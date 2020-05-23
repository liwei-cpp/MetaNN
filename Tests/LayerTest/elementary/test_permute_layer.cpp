#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <data_gen.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
    void test_permute_layer1()
    {
        cout << "Test permute layer case 1 ...\t";
        using RootLayer = MakeInferLayer<PermuteLayer, PDimArrayIs<2, 0, 1>>;
        static_assert(!RootLayer::IsFeedbackOutput);
        static_assert(!RootLayer::IsUpdate);

        RootLayer layer("root");

        auto in = GenTensor<CheckElement>(-3.3f, 0.1f, 7, 4, 5);
        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
    
        assert(res.Shape() == Shape(5, 7, 4));

        for (size_t i = 0; i < 7; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                for (size_t k = 0; k < 5; ++k)
                {
                    assert(fabs(res(k, i, j) - in(i, j, k)) < 0.0001);
                }
            }
        }

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<LayerInput>);

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }

    using InputMap1 = LayerInMap<LayerKV<LayerInput, Tensor<CheckElement, CheckDevice, 3>>>;
    void test_permute_layer2()
    {
        cout << "Test permute layer case 2 ...\t";
        using RootLayer = MakeTrainLayer<PermuteLayer, InputMap1,
                                         PDimArrayIs<2, 0, 1>, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput);
        static_assert(!RootLayer::IsUpdate);

        RootLayer layer("root");

        auto in = GenTensor<CheckElement>(-3.3f, 0.1f, 7, 4, 5);
        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
    
        assert(res.Shape() == Shape(5, 7, 4));

        for (size_t i = 0; i < 7; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                for (size_t k = 0; k < 5; ++k)
                {
                    assert(fabs(res(k, i, j) - in(i, j, k)) < 0.0001);
                }
            }
        }

        auto grad = GenTensor<CheckElement>(0, 1, 5, 7, 4);
        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));
        auto fb = Evaluate(out_grad.Get<LayerInput>());

        for (size_t i = 0; i < 5; ++i)
        {
            for (size_t j = 0; j < 7; ++j)
            {
                for (size_t k = 0; k < 4; ++k)
                {
                    assert(fabs(fb(j, k, i) - grad(i, j, k)) < 0.0001);
                }
            }
        }

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
    
    void test_permute_layer3()
    {
        cout << "Test permute layer case 2 (dummy grad) ...\t";
        using RootLayer = MakeTrainLayer<PermuteLayer, InputMap1,
                                         PDimArrayIs<2, 0, 1>, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput);
        static_assert(!RootLayer::IsUpdate);

        RootLayer layer("root");

        auto in = GenTensor<CheckElement>(-3.3f, 0.1f, 7, 4, 5);
        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
    
        assert(res.Shape() == Shape(5, 7, 4));

        for (size_t i = 0; i < 7; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                for (size_t k = 0; k < 5; ++k)
                {
                    assert(fabs(res(k, i, j) - in(i, j, k)) < 0.0001);
                }
            }
        }

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<LayerInput>);

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }

    void test_permute_layer4()
    {
        cout << "Test transpose layer case 1 ...\t";
        using RootLayer = MakeInferLayer<TransposeLayer>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        auto in = GenTensor<CheckElement>(-3.3f, 0.1f, 4, 5);
        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
    
        assert(res.Shape() == Shape(5, 4));
    
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                assert(fabs(res(j, i) - in(i, j)) < 0.0001);
            }
        }

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<LayerInput>);

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }

    using InputMap2 = LayerInMap<LayerKV<LayerInput, Matrix<CheckElement, CheckDevice>>>;
    void test_permute_layer5()
    {
        cout << "Test transpose layer case 2 ...\t";
        using RootLayer = MakeTrainLayer<TransposeLayer, InputMap2, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput);
        static_assert(!RootLayer::IsUpdate);

        RootLayer layer("root");

        auto in = GenTensor<CheckElement>(-3.3f, 0.1f, 4, 5);
        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(res.Shape() == Shape(5, 4));
    
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                assert(fabs(res(j, i) - in(i, j)) < 0.0001);
            }
        }

        auto grad = GenTensor<float>(1.8f, -0.2f, 5, 4);
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
    
    void test_permute_layer6()
    {
        cout << "Test transpose layer case 3 (dummy grad input)...\t";
        using RootLayer = MakeTrainLayer<TransposeLayer, InputMap2, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput);
        static_assert(!RootLayer::IsUpdate);

        RootLayer layer("root");

        auto in = GenTensor<CheckElement>(-3.3f, 0.1f, 4, 5);
        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(res.Shape() == Shape(5, 4));
    
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                assert(fabs(res(j, i) - in(i, j)) < 0.0001);
            }
        }

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<LayerInput>);
        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
}

namespace Test::Layer::Elementary
{
    void test_permute_layer()
    {
        test_permute_layer1();
        test_permute_layer2();
        test_permute_layer3();
        
        test_permute_layer4();
        test_permute_layer5();
        test_permute_layer6();
    }
}