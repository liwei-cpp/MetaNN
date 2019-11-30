#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <data_gen.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
    using CommonInputMap = LayerIOMap<LayerKV<InterpolateLayerWeight1, Matrix<CheckElement, CheckDevice>>,
                                      LayerKV<InterpolateLayerWeight2, Matrix<CheckElement, CheckDevice>>,
                                      LayerKV<InterpolateLayerLambda, Matrix<CheckElement, CheckDevice>>>;
    void test_interpolate_layer1()
    {
        cout << "Test interpolate layer case 1 ...\t";
        using RootLayer = MakeInferLayer<InterpolateLayer>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        Matrix<CheckElement, CheckDevice> i1(2, 3);
        i1.SetValue(0, 0, 0.1f);  i1.SetValue(0, 1, 0.2f); i1.SetValue(0, 2, 0.3f);
        i1.SetValue(1, 0, 0.4f);  i1.SetValue(1, 1, 0.5f); i1.SetValue(1, 2, 0.6f);

        Matrix<CheckElement, CheckDevice> i2(2, 3);
        i2.SetValue(0, 0, 0.2f);  i2.SetValue(0, 1, 0.3f); i2.SetValue(0, 2, 0.4f);
        i2.SetValue(1, 0, 0.5f);  i2.SetValue(1, 1, 0.6f); i2.SetValue(1, 2, 0.7f);

        Matrix<CheckElement, CheckDevice> delta(2, 3);
        delta.SetValue(0, 0, 0.3f);  delta.SetValue(0, 1, 0.6f); delta.SetValue(0, 2, 0.9f);
        delta.SetValue(1, 0, 0.4f);  delta.SetValue(1, 1, 0.1f); delta.SetValue(1, 2, 0.7f);

        auto input = LayerInputCont<RootLayer>().Set<InterpolateLayerWeight1>(i1)
                                                .Set<InterpolateLayerWeight2>(i2)
                                                .Set<InterpolateLayerLambda>(delta);

        LayerNeutralInvariant(layer);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(fabs(res(0, 0) - 0.17f) < 0.001);
        assert(fabs(res(0, 1) - 0.24f) < 0.001);
        assert(fabs(res(0, 2) - 0.31f) < 0.001);
        assert(fabs(res(1, 0) - 0.46f) < 0.001);
        assert(fabs(res(1, 1) - 0.59f) < 0.001);
        assert(fabs(res(1, 2) - 0.63f) < 0.001);

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<InterpolateLayerWeight1>);
        static_assert(decltype(out_grad)::template IsValueEmpty<InterpolateLayerWeight2>);
        static_assert(decltype(out_grad)::template IsValueEmpty<InterpolateLayerLambda>);

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }

    void test_interpolate_layer2()
    {
        cout << "Test interpolate layer case 2 ...\t";
        using RootLayer = MakeTrainLayer<InterpolateLayer, CommonInputMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput);
        static_assert(!RootLayer::IsUpdate);

        RootLayer layer("root");

        Matrix<CheckElement, CheckDevice> i1(2, 3);
        i1.SetValue(0, 0, 0.1f);  i1.SetValue(0, 1, 0.2f); i1.SetValue(0, 2, 0.3f);
        i1.SetValue(1, 0, 0.4f);  i1.SetValue(1, 1, 0.5f); i1.SetValue(1, 2, 0.6f);

        Matrix<CheckElement, CheckDevice> i2(2, 3);
        i2.SetValue(0, 0, 0.2f);  i2.SetValue(0, 1, 0.3f); i2.SetValue(0, 2, 0.4f);
        i2.SetValue(1, 0, 0.5f);  i2.SetValue(1, 1, 0.6f); i2.SetValue(1, 2, 0.7f);

        Matrix<CheckElement, CheckDevice> delta(2, 3);
        delta.SetValue(0, 0, 0.3f);  delta.SetValue(0, 1, 0.6f); delta.SetValue(0, 2, 0.9f);
        delta.SetValue(1, 0, 0.4f);  delta.SetValue(1, 1, 0.1f); delta.SetValue(1, 2, 0.7f);

        auto input = LayerInputCont<RootLayer>().Set<InterpolateLayerWeight1>(i1)
                                                .Set<InterpolateLayerWeight2>(i2)
                                                .Set<InterpolateLayerLambda>(delta);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(fabs(res(0, 0) - 0.17f) < 0.001);
        assert(fabs(res(0, 1) - 0.24f) < 0.001);
        assert(fabs(res(0, 2) - 0.31f) < 0.001);
        assert(fabs(res(1, 0) - 0.46f) < 0.001);
        assert(fabs(res(1, 1) - 0.59f) < 0.001);
        assert(fabs(res(1, 2) - 0.63f) < 0.001);

        Matrix<CheckElement, CheckDevice> grad(2, 3);
        grad.SetValue(0, 0, 0.2f);  grad.SetValue(0, 1, 0.5f); grad.SetValue(0, 2, 0.8f);
        grad.SetValue(1, 0, 0.7f);  grad.SetValue(1, 1, 0.6f); grad.SetValue(1, 2, 0.3f);

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));

        auto r = Evaluate(out_grad.Get<InterpolateLayerWeight1>());
        assert(fabs(r(0, 0) - 0.06f) < 0.001);
        assert(fabs(r(0, 1) - 0.30f) < 0.001);
        assert(fabs(r(0, 2) - 0.72f) < 0.001);
        assert(fabs(r(1, 0) - 0.28f) < 0.001);
        assert(fabs(r(1, 1) - 0.06f) < 0.001);
        assert(fabs(r(1, 2) - 0.21f) < 0.001);

        r = Evaluate(out_grad.Get<InterpolateLayerWeight2>());
        assert(fabs(r(0, 0) - 0.14f) < 0.001);
        assert(fabs(r(0, 1) - 0.20f) < 0.001);
        assert(fabs(r(0, 2) - 0.08f) < 0.001);
        assert(fabs(r(1, 0) - 0.42f) < 0.001);
        assert(fabs(r(1, 1) - 0.54f) < 0.001);
        assert(fabs(r(1, 2) - 0.09f) < 0.001);

        r = Evaluate(out_grad.Get<InterpolateLayerLambda>());
        assert(fabs(r(0, 0) + 0.02f) < 0.001);
        assert(fabs(r(0, 1) + 0.05f) < 0.001);
        assert(fabs(r(0, 2) + 0.08f) < 0.001);
        assert(fabs(r(1, 0) + 0.07f) < 0.001);
        assert(fabs(r(1, 1) + 0.06f) < 0.001);
        assert(fabs(r(1, 2) + 0.03f) < 0.001);

        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }

    void test_interpolate_layer3()
    {
        cout << "Test interpolate layer case 3 ...\t";
        using RootLayer = MakeTrainLayer<InterpolateLayer, CommonInputMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        vector<Matrix<CheckElement, CheckDevice>> op1;
        vector<Matrix<CheckElement, CheckDevice>> op2;
        vector<Matrix<CheckElement, CheckDevice>> opdelta;

        LayerNeutralInvariant(layer);
        for (size_t loop_count = 1; loop_count < 10; ++loop_count)
        {
            auto i1 = GenMatrix<CheckElement>(loop_count, 3, 0.1f, 0.13f);
            auto i2 = GenMatrix<CheckElement>(loop_count, 3, -0.2f, 0.05f);
            auto delta = GenMatrix<CheckElement>(loop_count, 3, 1.2f, 0.07f);

            op1.push_back(i1);
            op2.push_back(i2);
            opdelta.push_back(delta);

            auto input = LayerInputCont<RootLayer>().Set<InterpolateLayerWeight1>(i1)
                                                    .Set<InterpolateLayerWeight2>(i2)
                                                    .Set<InterpolateLayerLambda>(delta);

            auto out = layer.FeedForward(input);
            auto res = Evaluate(out.Get<LayerOutput>());
            assert(res.Shape().RowNum() == loop_count);
            assert(res.Shape().ColNum() == 3);
            for (size_t i = 0; i < loop_count; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    CheckElement aim = i1(i, j) * delta(i, j);
                    aim += i2(i, j) * (1 - delta(i, j));
                    assert(fabs(res(i, j) - aim) < 0.0001);
                }
            }
        }

        for (size_t loop_count = 9; loop_count >= 1; --loop_count)
        {
            auto grad = GenMatrix<CheckElement>(loop_count, 3, 2, 1.1f);
            auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));

            auto handle1 = out_grad.Get<InterpolateLayerWeight1>().EvalRegister();
            auto handle2 = out_grad.Get<InterpolateLayerWeight2>().EvalRegister();
            auto handle3 = out_grad.Get<InterpolateLayerLambda>().EvalRegister();
            EvalPlan<DeviceTags::CPU>::Eval();

            auto r1 = handle1.Data();
            auto r2 = handle2.Data();
            auto rlambda = handle3.Data();

            auto i1 = op1.back(); op1.pop_back();
            auto i2 = op2.back(); op2.pop_back();
            auto delta = opdelta.back(); opdelta.pop_back();
            for (size_t i = 0; i < loop_count; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    assert(fabs(r1(i, j) - grad(i, j) * delta(i, j)) < 0.00001f);
                    assert(fabs(r2(i, j) - grad(i, j) * (1 - delta(i, j))) < 0.00001f);
                    assert(fabs(rlambda(i, j) - grad(i, j) * (i1(i, j) - i2(i, j))) < 0.00001f);
                }
            }
        }

        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }
    
    void test_interpolate_layer4()
    {
        cout << "Test interpolate layer case 4 (dummy grad input)...\t";
        using RootLayer = MakeTrainLayer<InterpolateLayer, CommonInputMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput);
        static_assert(!RootLayer::IsUpdate);

        RootLayer layer("root");

        Matrix<CheckElement, CheckDevice> i1(2, 3);
        i1.SetValue(0, 0, 0.1f);  i1.SetValue(0, 1, 0.2f); i1.SetValue(0, 2, 0.3f);
        i1.SetValue(1, 0, 0.4f);  i1.SetValue(1, 1, 0.5f); i1.SetValue(1, 2, 0.6f);

        Matrix<CheckElement, CheckDevice> i2(2, 3);
        i2.SetValue(0, 0, 0.2f);  i2.SetValue(0, 1, 0.3f); i2.SetValue(0, 2, 0.4f);
        i2.SetValue(1, 0, 0.5f);  i2.SetValue(1, 1, 0.6f); i2.SetValue(1, 2, 0.7f);

        Matrix<CheckElement, CheckDevice> delta(2, 3);
        delta.SetValue(0, 0, 0.3f);  delta.SetValue(0, 1, 0.6f); delta.SetValue(0, 2, 0.9f);
        delta.SetValue(1, 0, 0.4f);  delta.SetValue(1, 1, 0.1f); delta.SetValue(1, 2, 0.7f);

        auto input = LayerInputCont<RootLayer>().Set<InterpolateLayerWeight1>(i1)
                                                .Set<InterpolateLayerWeight2>(i2)
                                                .Set<InterpolateLayerLambda>(delta);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(fabs(res(0, 0) - 0.17f) < 0.001);
        assert(fabs(res(0, 1) - 0.24f) < 0.001);
        assert(fabs(res(0, 2) - 0.31f) < 0.001);
        assert(fabs(res(1, 0) - 0.46f) < 0.001);
        assert(fabs(res(1, 1) - 0.59f) < 0.001);
        assert(fabs(res(1, 2) - 0.63f) < 0.001);

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<InterpolateLayerWeight1>);
        static_assert(decltype(out_grad)::template IsValueEmpty<InterpolateLayerWeight2>);
        static_assert(decltype(out_grad)::template IsValueEmpty<InterpolateLayerLambda>);

        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }
}
namespace Test::Layer::Elementary
{
    void test_interpolate_layer()
    {
        test_interpolate_layer1();
        test_interpolate_layer2();
        test_interpolate_layer3();
        test_interpolate_layer4();
    }
}