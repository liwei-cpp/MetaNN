#include <MetaNN/meta_nn.h>
#include "../../facilities/data_gen.h"
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_negative_log_likelihood_layer1()
{
    cout << "Test negative log likelyhood layer case 1 ...\t";
    using RootLayer = InjectPolicy<NegativeLogLikelihoodLayer>;
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;
    auto in = GenMatrix<float>(3, 4, 0.1f, 0.05f);
    auto label = GenMatrix<float>(3, 4, 0.3f, 0.1f);

    auto input = CostLayerIn::Create().Set<CostLayerIn>(in).Set<CostLayerLabel>(label);

    LayerNeutralInvariant(layer);

    auto out = layer.FeedForward(input);
    auto res = Evaluate(out.Get<LayerIO>());
    float check = 0;
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            check -= log(in(i, j)) * label(i, j);
        }
    }
    assert(fabs(res.Value() - check) < 0.0001);

    LayerNeutralInvariant(layer);

    NullParameter fbIn;
    auto out_grad = layer.FeedBackward(fbIn);
    auto fb1 = out_grad.Get<CostLayerIn>();
    static_assert(std::is_same<decltype(fb1), NullParameter>::value, "Test error");

    cout << "done" << endl;
}

void test_negative_log_likelihood_layer2()
{
    cout << "Test negative log likelyhood layer case 2 ...\t";
    using RootLayer = InjectPolicy<NegativeLogLikelihoodLayer, PFeedbackOutput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;
    auto in = GenMatrix<float>(3, 4, 0.1f, 0.05f);
    auto label = GenMatrix<float>(3, 4, 0.3f, 0.1f);

    auto input = CostLayerIn::Create()
                        .Set<CostLayerIn>(in)
                        .Set<CostLayerLabel>(label);

    LayerNeutralInvariant(layer);

    auto out = layer.FeedForward(input);
    auto res = Evaluate(out.Get<LayerIO>());
    float check = 0;
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            check -= log(in(i, j)) * label(i, j);
        }
    }
    assert(fabs(res.Value() - check) < 0.0001);

    auto fb = LayerIO::Create().Set<LayerIO>(Scalar<float>(0.5));
    auto out_grad = layer.FeedBackward(fb);
    LayerNeutralInvariant(layer);

    auto g = Evaluate(out_grad.Get<CostLayerIn>());
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            assert(fabs(g(i, j) + 0.5 * label(i, j) / in(i, j)) < 0.0001);
        }
    }
    cout << "done" << endl;
}

void test_negative_log_likelihood_layer3()
{
    cout << "Test negative log likelyhood layer case 3 ...\t";
    using RootLayer = InjectPolicy<NegativeLogLikelihoodLayer, PFeedbackOutput>;
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    static_assert(!RootLayer::IsUpdate, "Test Error");

    RootLayer layer;

    vector<Matrix<float, DeviceTags::CPU>> op_in;
    vector<Matrix<float, DeviceTags::CPU>> op_label;

    LayerNeutralInvariant(layer);
    for (size_t loop_count = 1; loop_count < 10; ++loop_count)
    {
        auto in = GenMatrix<float>(loop_count * 2, 4, 0.1f, 0.05f);
        auto label = GenMatrix<float>(loop_count * 2, 4, 0.3f, 0.1f);

        op_in.push_back(in);
        op_label.push_back(label);

        auto input = CostLayerIn::Create().Set<CostLayerIn>(in).Set<CostLayerLabel>(label);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerIO>());

        float check = 0;
        for (size_t i = 0; i < loop_count * 2; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                check -= log(in(i, j)) * label(i, j);
            }
        }
        assert(fabs(res.Value() - check) < 0.0001);
    }

    for (size_t loop_count = 9; loop_count >= 1; --loop_count)
    {
        auto out_grad = layer.FeedBackward(LayerIO::Create().Set<LayerIO>(Scalar<float>(0.5 * loop_count)));
        auto fb = Evaluate(out_grad.Get<CostLayerIn>());

        auto in = op_in.back(); op_in.pop_back();
        auto label = op_label.back(); op_label.pop_back();
        for (size_t i = 0; i < loop_count * 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(fb(i, j) + 0.5 * loop_count * label(i, j) / in(i, j)) < 0.0001);
            }
        }
    }

    LayerNeutralInvariant(layer);
    cout << "done" << endl;
}
}

void test_negative_log_likelihood_layer()
{
    test_negative_log_likelihood_layer1();
    test_negative_log_likelihood_layer2();
    test_negative_log_likelihood_layer3();
}
