#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <data_gen.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
    using CommonInputMap = LayerIOMap<LayerKV<LayerInput, Matrix<CheckElement, CheckDevice>>,
                                      LayerKV<LossLayerWeight, Matrix<CheckElement, CheckDevice>>>;

    void test_nll_loss_layer1()
    {
        cout << "Test NLL loss layer case 1 ...\t";
        using RootLayer = MakeInferLayer<NLLLossLayer>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");
        auto in = GenMatrix<CheckElement>(3, 4, 0.1f, 0.05f);
        auto weight = GenMatrix<CheckElement>(3, 4, 0.3f, 0.1f);

        auto input = LayerInputCont<RootLayer>()
                        .Set<LayerInput>(in)
                        .Set<LossLayerWeight>(weight);

        LayerNeutralInvariant(layer);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());

        CheckElement check = 0;
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                check -= log(in(i, j)) * weight(i, j);
            }
        }
        assert(fabs(res.Value() - check) < 0.0001);

        LayerNeutralInvariant(layer);

        NullParameter fbIn;
        auto out_grad = layer.FeedBackward(fbIn);
        static_assert(decltype(out_grad)::template IsValueEmpty<LayerInput>);

        cout << "done" << endl;
    }

    void test_nll_loss_layer2()
    {
        cout << "Test NLL loss layer case 2 ...\t";
        using RootLayer = MakeTrainLayer<NLLLossLayer, CommonInputMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");
        auto in = GenMatrix<CheckElement>(3, 4, 0.1f, 0.05f);
        auto weight = GenMatrix<CheckElement>(3, 4, 0.3f, 0.1f);

        auto input = LayerInputCont<RootLayer>()
                        .Set<LayerInput>(in)
                        .Set<LossLayerWeight>(weight);

        LayerNeutralInvariant(layer);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        CheckElement check = 0;
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                check -= log(in(i, j)) * weight(i, j);
            }
        }
        assert(fabs(res.Value() - check) < 0.0001);

        auto fb = LayerOutputCont<RootLayer>().Set<LayerOutput>(Scalar<CheckElement, CheckDevice>(0.5));
        auto out_grad = layer.FeedBackward(fb);
        LayerNeutralInvariant(layer);

        auto g = Evaluate(out_grad.Get<LayerInput>());
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                assert(fabs(g(i, j) + 0.5 * weight(i, j) / in(i, j)) < 0.0001);
            }
        }
        cout << "done" << endl;
    }
    
    void test_nll_loss_layer3()
    {
        cout << "Test NLL loss layer case 3 ...\t";
        using RootLayer = MakeTrainLayer<NLLLossLayer, CommonInputMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        vector<Matrix<CheckElement, CheckDevice>> op_in;
        vector<Matrix<CheckElement, CheckDevice>> op_weight;

        LayerNeutralInvariant(layer);
        for (size_t loop_count = 1; loop_count < 10; ++loop_count)
        {
            auto in = GenMatrix<CheckElement>(loop_count * 2, 4, 0.1f, 0.05f);
            auto weight = GenMatrix<CheckElement>(loop_count * 2, 4, 0.3f, 0.1f);

            op_in.push_back(in);
            op_weight.push_back(weight);

            auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in).Set<LossLayerWeight>(weight);

            auto out = layer.FeedForward(input);
            auto res = Evaluate(out.Get<LayerOutput>());

            CheckElement check = 0;
            for (size_t i = 0; i < loop_count * 2; ++i)
            {
                for (size_t j = 0; j < 4; ++j)
                {
                    check -= log(in(i, j)) * weight(i, j);
                }
            }
            assert(fabs(res.Value() - check) < 0.0001);
        }

        for (size_t loop_count = 9; loop_count >= 1; --loop_count)
        {
            auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(Scalar<CheckElement, CheckDevice>(0.5 * loop_count)));
            auto fb = Evaluate(out_grad.Get<LayerInput>());

            auto in = op_in.back(); op_in.pop_back();
            auto weight = op_weight.back(); op_weight.pop_back();
            for (size_t i = 0; i < loop_count * 2; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    assert(fabs(fb(i, j) + 0.5 * loop_count * weight(i, j) / in(i, j)) < 0.0001);
                }
            }
        }

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
}

namespace Test::Layer::Loss
{
    void test_nll_loss_layer()
    {
        test_nll_loss_layer1();
        test_nll_loss_layer2();
        test_nll_loss_layer3();
    }
}