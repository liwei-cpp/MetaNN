#include <MetaNN/meta_nn.h>
#include "../../facilities/data_gen.h"
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_add_layer1()
{
    cout << "Test add layer case 1 ...\t";
    using RootLayer = InjectPolicy<AddLayer>;
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    auto i1 = GenMatrix<float>(2, 3, 1, 0.1f);
    auto i2 = GenMatrix<float>(2, 3, 1.5f, -0.1f);

    auto input = AddLayerInput::Create().Set<AddLayerIn1>(i1)
                                        .Set<AddLayerIn2>(i2);

    auto out = layer.FeedForward(input);
    auto res = Evaluate(out.Get<LayerIO>());
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            assert(fabs(res(i, j) - i1(i, j) - i2(i, j)) < 0.001);
        }
    }

    NullParameter fbIn;
    auto out_grad = layer.FeedBackward(fbIn);
    auto fb1 = out_grad.Get<AddLayerIn1>();
    auto fb2 = out_grad.Get<AddLayerIn2>();
    static_assert(std::is_same<decltype(fb1), NullParameter>::value, "Test error");
    static_assert(std::is_same<decltype(fb2), NullParameter>::value, "Test error");
    cout << "done" << endl;
}

void test_add_layer2()
{
    cout << "Test add layer case 2 ...\t";

    using RootLayer = InjectPolicy<AddLayer, PFeedbackOutput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;
    auto i1 = GenMatrix<float>(2, 3, 1, 0.1f);
    auto i2 = GenMatrix<float>(2, 3, 1.5f, -0.1f);

    auto input = AddLayerInput::Create().Set<AddLayerIn1>(i1)
                                        .Set<AddLayerIn2>(i2);

    auto out = layer.FeedForward(input);
    auto res = Evaluate(out.Get<LayerIO>());
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            assert(fabs(res(i, j) - i1(i, j) - i2(i, j)) < 0.001);
        }
    }

    auto grad = GenMatrix<float>(2, 3, 0.7f, -0.2f);

    auto out_grad = layer.FeedBackward(RootLayer::OutputType::Create().Set<LayerIO>(grad));

    auto handle1 = out_grad.Get<AddLayerIn1>().EvalRegister();
    auto handle2 = out_grad.Get<AddLayerIn2>().EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();

    auto fb1 = handle1.Data();
    auto fb2 = handle2.Data();
    assert(fb1.RowNum() == fb2.RowNum());
    assert(fb1.ColNum() == fb2.ColNum());
    assert(fb1.RowNum() == 2);
    assert(fb1.ColNum() == 3);

    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            assert(fb1(i, j) == grad(i, j));
            assert(fb2(i, j) == grad(i, j));
        }
    }

    cout << "done" << endl;
}

void test_add_layer3()
{
    cout << "Test add layer case 3 ...\t";
    using RootLayer = InjectPolicy<AddLayer, PFeedbackOutput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    auto i1 = GenMatrix<float>(2, 3, 1, 0.1f);
    auto i2 = GenMatrix<float>(2, 3, 1.5f, -0.1f);

    auto input = AddLayerInput::Create().Set<AddLayerIn1>(i1)
                                        .Set<AddLayerIn2>(i2);

    auto out = layer.FeedForward(input);
    auto res = Evaluate(out.Get<LayerIO>());
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            assert(fabs(res(i, j) - i1(i, j) - i2(i, j)) < 0.001);
        }
    }

    auto i3 = GenMatrix<float>(2, 3, 1.3, -0.1f);
    auto i4 = GenMatrix<float>(2, 3, 2.5f, -0.7f);

    input = AddLayerInput::Create().Set<AddLayerIn1>(i3).Set<AddLayerIn2>(i4);

    out = layer.FeedForward(input);
    res = Evaluate(out.Get<LayerIO>());
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            assert(fabs(res(i, j) - i3(i, j) - i4(i, j)) < 0.001);
        }
    }

    auto grad = GenMatrix<float>(2, 3, 0.7f, -0.2f);

    auto out_grad = layer.FeedBackward(RootLayer::OutputType::Create().Set<LayerIO>(grad));

    auto handle1 = out_grad.Get<AddLayerIn1>().EvalRegister();
    auto handle2 = out_grad.Get<AddLayerIn2>().EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();

    auto fb1 = handle1.Data();
    auto fb2 = handle2.Data();
    assert(fb1.RowNum() == fb2.RowNum());
    assert(fb1.ColNum() == fb2.ColNum());
    assert(fb1.RowNum() == 2);
    assert(fb1.ColNum() == 3);

    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            assert(fb1(i, j) == grad(i, j));
            assert(fb2(i, j) == grad(i, j));
        }
    }

    grad = GenMatrix<float>(2, 3, -0.7f, 0.2f);

    out_grad = layer.FeedBackward(RootLayer::OutputType::Create().Set<LayerIO>(grad));

    handle1 = out_grad.Get<AddLayerIn1>().EvalRegister();
    handle2 = out_grad.Get<AddLayerIn2>().EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();

    fb1 = handle1.Data();
    fb2 = handle2.Data();

    assert(fb1.RowNum() == fb2.RowNum());
    assert(fb1.ColNum() == fb2.ColNum());
    assert(fb1.RowNum() == 2);
    assert(fb1.ColNum() == 3);

    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            assert(fb1(i, j) == grad(i, j));
            assert(fb2(i, j) == grad(i, j));
        }
    }
    cout << "done" << endl;
}
}

void test_add_layer()
{
    test_add_layer1();
    test_add_layer2();
    test_add_layer3();
}
