#include <MetaNN/meta_nn.h>
#include "../../facilities/data_gen.h"
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_element_mul_layer1()
{
    cout << "Test element mul layer case 1 ...\t";
    using RootLayer = InjectPolicy<ElementMulLayer>;
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    Matrix<float, DeviceTags::CPU> i1(2, 3);
    i1.SetValue(0, 0, 0.1f);  i1.SetValue(0, 1, 0.2f); i1.SetValue(0, 2, 0.3f);
    i1.SetValue(1, 0, 0.4f);  i1.SetValue(1, 1, 0.5f); i1.SetValue(1, 2, 0.6f);

    Matrix<float, DeviceTags::CPU> i2(2, 3);
    i2.SetValue(0, 0, 0.2f);  i2.SetValue(0, 1, 0.3f); i2.SetValue(0, 2, 0.4f);
    i2.SetValue(1, 0, 0.5f);  i2.SetValue(1, 1, 0.6f); i2.SetValue(1, 2, 0.7f);

    auto input = ElementMulLayerInput::Create().Set<ElementMulLayerIn1>(i1)
                                               .Set<ElementMulLayerIn2>(i2);

    LayerNeutralInvariant(layer);

    auto out = layer.FeedForward(input);
    auto res = Evaluate(out.Get<LayerIO>());
    assert(fabs(res(0, 0) - 0.02f) < 0.001);
    assert(fabs(res(0, 1) - 0.06f) < 0.001);
    assert(fabs(res(0, 2) - 0.12f) < 0.001);
    assert(fabs(res(1, 0) - 0.20f) < 0.001);
    assert(fabs(res(1, 1) - 0.30f) < 0.001);
    assert(fabs(res(1, 2) - 0.42f) < 0.001);

    auto out_grad = layer.FeedBackward(LayerIO::Create());
    auto fb1 = out_grad.Get<ElementMulLayerIn1>();
    auto fb2 = out_grad.Get<ElementMulLayerIn2>();
    static_assert(std::is_same<decltype(fb1), NullParameter>::value, "Test error");
    static_assert(std::is_same<decltype(fb2), NullParameter>::value, "Test error");

    LayerNeutralInvariant(layer);
    cout << "done" << endl;
}

void test_element_mul_layer2()
{
    cout << "Test element mul layer case 2 ...\t";
    using RootLayer = InjectPolicy<ElementMulLayer, PFeedbackOutput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    Matrix<float, DeviceTags::CPU> i1(2, 3);
    i1.SetValue(0, 0, 0.1f);  i1.SetValue(0, 1, 0.2f); i1.SetValue(0, 2, 0.3f);
    i1.SetValue(1, 0, 0.4f);  i1.SetValue(1, 1, 0.5f); i1.SetValue(1, 2, 0.6f);

    Matrix<float, DeviceTags::CPU> i2(2, 3);
    i2.SetValue(0, 0, 0.2f);  i2.SetValue(0, 1, 0.3f); i2.SetValue(0, 2, 0.4f);
    i2.SetValue(1, 0, 0.5f);  i2.SetValue(1, 1, 0.6f); i2.SetValue(1, 2, 0.7f);

    auto input = ElementMulLayerInput::Create().Set<ElementMulLayerIn1>(i1)
                                               .Set<ElementMulLayerIn2>(i2);

    LayerNeutralInvariant(layer);

    auto out = layer.FeedForward(input);

    Matrix<float, DeviceTags::CPU> grad(2, 3);
    grad.SetValue(0, 0, 0.3f);  grad.SetValue(0, 1, 0.6f); grad.SetValue(0, 2, 0.9f);
    grad.SetValue(1, 0, 0.4f);  grad.SetValue(1, 1, 0.1f); grad.SetValue(1, 2, 0.7f);
    auto out_grad = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(grad));

    auto handle1 = out.Get<LayerIO>().EvalRegister();
    auto handle2 = out_grad.Get<ElementMulLayerIn1>().EvalRegister();
    auto handle3 = out_grad.Get<ElementMulLayerIn2>().EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();

    auto res = handle1.Data();
    assert(fabs(res(0, 0) - 0.02f) < 0.001);
    assert(fabs(res(0, 1) - 0.06f) < 0.001);
    assert(fabs(res(0, 2) - 0.12f) < 0.001);
    assert(fabs(res(1, 0) - 0.20f) < 0.001);
    assert(fabs(res(1, 1) - 0.30f) < 0.001);
    assert(fabs(res(1, 2) - 0.42f) < 0.001);

    auto g1 = handle2.Data();
    auto g2 = handle3.Data();
    assert(fabs(g1(0, 0) - 0.06f) < 0.001);
    assert(fabs(g1(0, 1) - 0.18f) < 0.001);
    assert(fabs(g1(0, 2) - 0.36f) < 0.001);
    assert(fabs(g1(1, 0) - 0.20f) < 0.001);
    assert(fabs(g1(1, 1) - 0.06f) < 0.001);
    assert(fabs(g1(1, 2) - 0.49f) < 0.001);

    assert(fabs(g2(0, 0) - 0.03f) < 0.001);
    assert(fabs(g2(0, 1) - 0.12f) < 0.001);
    assert(fabs(g2(0, 2) - 0.27f) < 0.001);
    assert(fabs(g2(1, 0) - 0.16f) < 0.001);
    assert(fabs(g2(1, 1) - 0.05f) < 0.001);
    assert(fabs(g2(1, 2) - 0.42f) < 0.001);

    LayerNeutralInvariant(layer);

    cout << "done" << endl;
}

void test_element_mul_layer3()
{
    cout << "Test element mul layer case 3 ...\t";
    using RootLayer = InjectPolicy<ElementMulLayer, PFeedbackOutput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    vector<Matrix<float, DeviceTags::CPU>> op1;
    vector<Matrix<float, DeviceTags::CPU>> op2;
    LayerNeutralInvariant(layer);
    for (size_t loop_count = 1; loop_count < 10; ++loop_count)
    {
        auto i1 = GenMatrix<float>(loop_count, 3, 0, 0.3f);
        auto i2 = GenMatrix<float>(loop_count, 3, -1, 1.3f);
        op1.push_back(i1);
        op2.push_back(i2);

        auto input = ElementMulLayerInput::Create().Set<ElementMulLayerIn1>(i1)
                                                   .Set<ElementMulLayerIn2>(i2);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerIO>());
        assert(res.RowNum() == loop_count);
        assert(res.ColNum() == 3);
        for (size_t i = 0; i < loop_count; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(res(i, j) - i1(i, j) * i2(i, j)) < 0.0001);
            }
        }
    }

    for (size_t loop_count = 9; loop_count >= 1; --loop_count)
    {
        auto grad = GenMatrix<float>(loop_count, 3, 2, 1.1f);
        auto out_grad = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(grad));

        auto handle1 = out_grad.Get<ElementMulLayerIn1>().EvalRegister();
        auto handle2 = out_grad.Get<ElementMulLayerIn2>().EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto g1 = handle1.Data();
        auto g2 = handle2.Data();

        auto i1 = op1.back(); op1.pop_back();
        auto i2 = op2.back(); op2.pop_back();
        for (size_t i = 0; i < loop_count; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(g1(i, j) - grad(i, j) * i2(i, j)) < 0.001);
                assert(fabs(g2(i, j) - grad(i, j) * i1(i, j)) < 0.001);
            }
        }
    }

    LayerNeutralInvariant(layer);

    cout << "done" << endl;
}
}

void test_element_mul_layer()
{
    test_element_mul_layer1();
    test_element_mul_layer2();
    test_element_mul_layer3();
}
