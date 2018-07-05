#include <MetaNN/meta_nn.h>
#include "../../facilities/data_gen.h"
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_interpolate_layer1()
{
    cout << "Test interpolate layer case 1 ...\t";
    using RootLayer = InjectPolicy<InterpolateLayer>;
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    Matrix<float, DeviceTags::CPU> i1(2, 3);
    i1.SetValue(0, 0, 0.1f);  i1.SetValue(0, 1, 0.2f); i1.SetValue(0, 2, 0.3f);
    i1.SetValue(1, 0, 0.4f);  i1.SetValue(1, 1, 0.5f); i1.SetValue(1, 2, 0.6f);

    Matrix<float, DeviceTags::CPU> i2(2, 3);
    i2.SetValue(0, 0, 0.2f);  i2.SetValue(0, 1, 0.3f); i2.SetValue(0, 2, 0.4f);
    i2.SetValue(1, 0, 0.5f);  i2.SetValue(1, 1, 0.6f); i2.SetValue(1, 2, 0.7f);

    Matrix<float, DeviceTags::CPU> delta(2, 3);
    delta.SetValue(0, 0, 0.3f);  delta.SetValue(0, 1, 0.6f); delta.SetValue(0, 2, 0.9f);
    delta.SetValue(1, 0, 0.4f);  delta.SetValue(1, 1, 0.1f); delta.SetValue(1, 2, 0.7f);

    auto input = InterpolateLayerInput::Create().Set<InterpolateLayerWeight1>(i1)
                                                .Set<InterpolateLayerWeight2>(i2)
                                                .Set<InterpolateLayerLambda>(delta);

    LayerNeutralInvariant(layer);

    auto out = layer.FeedForward(input);
    auto res = Evaluate(out.Get<LayerIO>());
    assert(fabs(res(0, 0) - 0.17f) < 0.001);
    assert(fabs(res(0, 1) - 0.24f) < 0.001);
    assert(fabs(res(0, 2) - 0.31f) < 0.001);
    assert(fabs(res(1, 0) - 0.46f) < 0.001);
    assert(fabs(res(1, 1) - 0.59f) < 0.001);
    assert(fabs(res(1, 2) - 0.63f) < 0.001);

    NullParameter fbIn;
    auto out_grad = layer.FeedBackward(fbIn);
    auto fb1 = out_grad.Get<InterpolateLayerWeight1>();
    auto fb2 = out_grad.Get<InterpolateLayerWeight2>();
    auto fb_lambda = out_grad.Get<InterpolateLayerLambda>();
    static_assert(std::is_same<decltype(fb1), NullParameter>::value, "Test error");
    static_assert(std::is_same<decltype(fb2), NullParameter>::value, "Test error");
    static_assert(std::is_same<decltype(fb_lambda), NullParameter>::value, "Test error");

    LayerNeutralInvariant(layer);
    cout << "done" << endl;
}

void test_interpolate_layer2()
{
    cout << "Test interpolate layer case 2 ...\t";
    using RootLayer = InjectPolicy<InterpolateLayer, PFeedbackOutput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    Matrix<float, DeviceTags::CPU> i1(2, 3);
    i1.SetValue(0, 0, 0.1f);  i1.SetValue(0, 1, 0.2f); i1.SetValue(0, 2, 0.3f);
    i1.SetValue(1, 0, 0.4f);  i1.SetValue(1, 1, 0.5f); i1.SetValue(1, 2, 0.6f);

    Matrix<float, DeviceTags::CPU> i2(2, 3);
    i2.SetValue(0, 0, 0.2f);  i2.SetValue(0, 1, 0.3f); i2.SetValue(0, 2, 0.4f);
    i2.SetValue(1, 0, 0.5f);  i2.SetValue(1, 1, 0.6f); i2.SetValue(1, 2, 0.7f);

    Matrix<float, DeviceTags::CPU> delta(2, 3);
    delta.SetValue(0, 0, 0.3f);  delta.SetValue(0, 1, 0.6f); delta.SetValue(0, 2, 0.9f);
    delta.SetValue(1, 0, 0.4f);  delta.SetValue(1, 1, 0.1f); delta.SetValue(1, 2, 0.7f);

    auto input = InterpolateLayerInput::Create().Set<InterpolateLayerWeight1>(i1)
                                                .Set<InterpolateLayerWeight2>(i2)
                                                .Set<InterpolateLayerLambda>(delta);

    LayerNeutralInvariant(layer);
    auto out = layer.FeedForward(input);
    auto res = Evaluate(out.Get<LayerIO>());
    assert(fabs(res(0, 0) - 0.17f) < 0.001);
    assert(fabs(res(0, 1) - 0.24f) < 0.001);
    assert(fabs(res(0, 2) - 0.31f) < 0.001);
    assert(fabs(res(1, 0) - 0.46f) < 0.001);
    assert(fabs(res(1, 1) - 0.59f) < 0.001);
    assert(fabs(res(1, 2) - 0.63f) < 0.001);

    Matrix<float, DeviceTags::CPU> grad(2, 3);
    grad.SetValue(0, 0, 0.2f);  grad.SetValue(0, 1, 0.5f); grad.SetValue(0, 2, 0.8f);
    grad.SetValue(1, 0, 0.7f);  grad.SetValue(1, 1, 0.6f); grad.SetValue(1, 2, 0.3f);

    auto out_grad = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(grad));

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
    using RootLayer = InjectPolicy<InterpolateLayer, PFeedbackOutput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    vector<Matrix<float, DeviceTags::CPU>> op1;
    vector<Matrix<float, DeviceTags::CPU>> op2;
    vector<Matrix<float, DeviceTags::CPU>> opdelta;

    LayerNeutralInvariant(layer);
    for (size_t loop_count = 1; loop_count < 10; ++loop_count)
    {
        auto i1 = GenMatrix<float>(loop_count, 3, 0.1f, 0.13f);
        auto i2 = GenMatrix<float>(loop_count, 3, -0.2f, 0.05f);
        auto delta = GenMatrix<float>(loop_count, 3, 1.2f, 0.07f);

        op1.push_back(i1);
        op2.push_back(i2);
        opdelta.push_back(delta);

        auto input = InterpolateLayerInput::Create().Set<InterpolateLayerWeight1>(i1)
                                                    .Set<InterpolateLayerWeight2>(i2)
                                                    .Set<InterpolateLayerLambda>(delta);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerIO>());
        assert(res.RowNum() == loop_count);
        assert(res.ColNum() == 3);
        for (size_t i = 0; i < loop_count; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                float aim = i1(i, j) * delta(i, j);
                aim += i2(i, j) * (1 - delta(i, j));
                assert(fabs(res(i, j) - aim) < 0.0001);
            }
        }
    }

    for (size_t loop_count = 9; loop_count >= 1; --loop_count)
    {
        auto grad = GenMatrix<float>(loop_count, 3, 2, 1.1f);
        auto out_grad = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(grad));

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
}

void test_interpolate_layer()
{
    test_interpolate_layer1();
    test_interpolate_layer2();
    test_interpolate_layer3();
}
