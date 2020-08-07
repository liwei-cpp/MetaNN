#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <data_gen.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
    using CommonInputMap = LayerInMap<LayerKV<LayerInput, Matrix<CheckElement, CheckDevice>>>;
    void test_abs_layer1()
    {
        cout << "Test abs layer case 1 ...\t";
        using RootLayer = MakeInferLayer<AbsLayer>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        auto in = GenTensor<CheckElement>(-3.3f, 0.1f, 4, 5);
        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
    
        assert(res.Shape()[0] == 4);
        assert(res.Shape()[1] == 5);
    
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                auto check = fabs(in(i, j));
                assert(fabs(res(i, j) - check) < 0.0001);
            }
        }

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<LayerInput>);

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }

    void test_abs_layer2()
    {
        cout << "Test abs layer case 2 ...\t";
        using RootLayer = MakeTrainLayer<AbsLayer, CommonInputMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        auto in = GenTensor<CheckElement>(-3.3f, 0.1f, 4, 5);
        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(res.Shape()[0] == 4);
        assert(res.Shape()[1] == 5);
    
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                auto check = fabs(in(i, j));
                assert(fabs(res(i, j) - check) < 0.0001);
            }
        }

        auto grad = GenTensor<float>(1.8f, -0.2f, 4, 5);
        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));
        auto fb = Evaluate(out_grad.Get<LayerInput>());
    
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                auto check = in(i, j) / fabs(in(i, j)) * grad(i, j);
                assert(fabs(fb(i, j) - check) < 0.0001);
            }
        }

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }

    void test_abs_layer3()
    {
        cout << "Test abs layer case 3 ...\t";
        using RootLayer = MakeTrainLayer<AbsLayer, CommonInputMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        vector<Matrix<CheckElement, CheckDevice>> op;

        LayerNeutralInvariant(layer);
        for (size_t loop_count = 1; loop_count < 10; ++loop_count)
        {
            auto in = GenTensor<CheckElement>(-0.1f, 0.02f, loop_count, 3);

            op.push_back(in);

            auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in);

            auto out = layer.FeedForward(input);
            auto res = Evaluate(out.Get<LayerOutput>());
            assert(res.Shape()[0] == loop_count);
            assert(res.Shape()[1] == 3);
            for (size_t i = 0; i < loop_count; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    auto check = fabs(in(i, j));
                    assert(fabs(res(i, j) - check) < 0.0001);
                }
            }
        }

        for (size_t loop_count = 9; loop_count >= 1; --loop_count)
        {
            auto grad = GenTensor<CheckElement>(2, 1.1f, loop_count, 3);
            auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));

            auto fb = Evaluate(out_grad.Get<LayerInput>());

            auto in = op.back(); op.pop_back();
            for (size_t i = 0; i < loop_count; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    float aim = 0;
                    if (in(i, j) > 0) aim = grad(i, j);
                    if (in(i, j) < 0) aim = -grad(i, j);
                    assert(fabs(fb(i, j) - aim) < 0.00001f);
                }
            }
        }

        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }

    void test_abs_layer4()
    {
        cout << "Test abs layer case 4 ...\t";
        using RootLayer = MakeTrainLayer<AbsLayer, CommonInputMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");
    
        Matrix<CheckElement, CheckDevice> x(1, 4);
        x.SetValue(0, 0, 0); x.SetValue(0, 1, -2); x.SetValue(0, 2, 3); x.SetValue(0, 3, -4);
        auto x_out = layer.FeedForward(LayerInputCont<RootLayer>().Set<LayerInput>(x));
        auto x_out_eval = Evaluate(x_out.Get<LayerOutput>());
        assert(fabs(x_out_eval(0, 0) - 0) <= 0.00001);
        assert(fabs(x_out_eval(0, 1) - 2) <= 0.00001);
        assert(fabs(x_out_eval(0, 2) - 3) <= 0.00001);
        assert(fabs(x_out_eval(0, 3) - 4) <= 0.00001);

        Matrix<float, DeviceTags::CPU> y(1, 4);
        y.SetValue(0, 0, 1); y.SetValue(0, 1, 5); y.SetValue(0, 2, 7); y.SetValue(0, 3, 3);
        auto y_out = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(y)).Get<LayerInput>();
        auto y_out_eval = Evaluate(y_out);
        assert(fabs(y_out_eval(0, 0) - 0) <= 0.00001);
        assert(fabs(y_out_eval(0, 1) + 5) <= 0.00001);
        assert(fabs(y_out_eval(0, 2) - 7) <= 0.00001);
        assert(fabs(y_out_eval(0, 3) + 3) <= 0.00001);
    
        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }
    
    void test_abs_layer5()
    {
        cout << "Test abs layer case 5 (dummy grad input)...\t";
        using RootLayer = MakeTrainLayer<AbsLayer, CommonInputMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        auto in = GenTensor<CheckElement>(-3.3f, 0.1f, 4, 5);
        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(res.Shape()[0] == 4);
        assert(res.Shape()[1] == 5);
    
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                auto check = fabs(in(i, j));
                assert(fabs(res(i, j) - check) < 0.0001);
            }
        }

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<LayerInput>);

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
}

namespace Test::Layer::Principal
{
    void test_abs_layer()
    {
        test_abs_layer1();
        test_abs_layer2();
        test_abs_layer3();
        test_abs_layer4();
        test_abs_layer5();
    }
}
