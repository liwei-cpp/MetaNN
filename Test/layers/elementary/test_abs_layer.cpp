#include <MetaNN/meta_nn.h>
#include "../../facilities/data_gen.h"
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_abs_layer1()
{
    cout << "Test abs layer case 1 ...\t";
    using RootLayer = InjectPolicy<AbsLayer>;
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    auto in = GenMatrix<float>(4, 5, -3.3f, 0.1f);
    auto input = LayerIO::Create().Set<LayerIO>(in);

    LayerNeutralInvariant(layer);
    auto out = layer.FeedForward(input);
    auto res = Evaluate(out.Get<LayerIO>());
    
    assert(res.RowNum() == 4);
    assert(res.ColNum() == 5);
    
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            auto check = fabs(in(i, j));
            assert(fabs(res(i, j) - check) < 0.0001);
        }
    }

    NullParameter fbIn;
    auto out_grad = layer.FeedBackward(fbIn);
    auto fb1 = out_grad.Get<LayerIO>();
    static_assert(std::is_same<decltype(fb1), NullParameter>::value, "Test error");

    LayerNeutralInvariant(layer);
    cout << "done" << endl;
}

void test_abs_layer2()
{
    cout << "Test abs layer case 2 ...\t";
    using RootLayer = InjectPolicy<AbsLayer, PFeedbackOutput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    auto in = GenMatrix<float>(4, 5, -3.3f, 0.1f);
    auto input = LayerIO::Create().Set<LayerIO>(in);

    LayerNeutralInvariant(layer);
    auto out = layer.FeedForward(input);
    auto res = Evaluate(out.Get<LayerIO>());
    assert(res.RowNum() == 4);
    assert(res.ColNum() == 5);
    
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 5; ++j)
        {
            auto check = fabs(in(i, j));
            assert(fabs(res(i, j) - check) < 0.0001);
        }
    }

    auto grad = GenMatrix<float>(4, 5, 1.8f, -0.2f);
    auto out_grad = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(grad));
    auto fb = Evaluate(out_grad.Get<LayerIO>());
    
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
    using RootLayer = InjectPolicy<AbsLayer, PFeedbackOutput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    vector<Matrix<float, DeviceTags::CPU>> op;

    LayerNeutralInvariant(layer);
    for (size_t loop_count = 1; loop_count < 10; ++loop_count)
    {
        auto in = GenMatrix<float>(loop_count, 3, -0.1f, 0.02f);

        op.push_back(in);

        auto input = LayerIO::Create().Set<LayerIO>(in);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerIO>());
        assert(res.RowNum() == loop_count);
        assert(res.ColNum() == 3);
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
        auto grad = GenMatrix<float>(loop_count, 3, 2, 1.1f);
        auto out_grad = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(grad));

        auto fb = Evaluate(out_grad.Get<LayerIO>());

        auto in = op.back(); op.pop_back();
        for (size_t i = 0; i < loop_count; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                auto aim = in(i, j) / fabs(in(i, j)) * grad(i, j);
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
    using RootLayer = InjectPolicy<AbsLayer, PFeedbackOutput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;
    
    Matrix<float, DeviceTags::CPU> x(1, 4);
    x.SetValue(0, 0, 0); x.SetValue(0, 1, -2); x.SetValue(0, 2, 3); x.SetValue(0, 3, -4);
    auto x_out = layer.FeedForward(LayerIO::Create().Set<LayerIO>(x));
    auto x_out_eval = Evaluate(x_out.Get<LayerIO>());
    assert(fabs(x_out_eval(0, 0) - 0) <= 0.00001);
    assert(fabs(x_out_eval(0, 1) - 2) <= 0.00001);
    assert(fabs(x_out_eval(0, 2) - 3) <= 0.00001);
    assert(fabs(x_out_eval(0, 3) - 4) <= 0.00001);

    Matrix<float, DeviceTags::CPU> y(1, 4);
    y.SetValue(0, 0, 1); y.SetValue(0, 1, 5); y.SetValue(0, 2, 7); y.SetValue(0, 3, 3);
    auto y_out = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(y)).Get<LayerIO>();
    auto y_out_eval = Evaluate(y_out);
    assert(fabs(y_out_eval(0, 0) - 0) <= 0.00001);
    assert(fabs(y_out_eval(0, 1) + 5) <= 0.00001);
    assert(fabs(y_out_eval(0, 2) - 7) <= 0.00001);
    assert(fabs(y_out_eval(0, 3) + 3) <= 0.00001);
    
    LayerNeutralInvariant(layer);

    cout << "done" << endl;
}
}

void test_abs_layer()
{
    test_abs_layer1();
    test_abs_layer2();
    test_abs_layer3();
    test_abs_layer4();
}
