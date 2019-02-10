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
    using CommonGradMap = LayerIOMap<LayerKV<LayerOutput, Matrix<CheckElement, CheckDevice>>>;
    
    void test_softmax_layer1()
    {
        cout << "Test softmax layer case 1 ...\t";
        using RootLayer = MakeBPLayer<SoftmaxLayer, CommonInputMap, CommonGradMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        Matrix<CheckElement, CheckDevice> in(1, 2);
        in.SetValue(0, 0, -0.27f);
        in.SetValue(0, 1, -0.41f);

        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in);

        LayerNeutralInvariant(layer);

        auto out = layer.FeedForward(input);
        auto check = Softmax(in);

        auto handle1 = out.Get<LayerOutput>().EvalRegister();
        auto handle2 = check.EvalRegister();
        EvalPlan<CheckDevice>::Eval();

        auto res = handle1.Data();
        auto c = handle2.Data();

        assert(fabs(res(0, 0) - c(0, 0)) < 0.001);
        assert(fabs(res(0, 1) - c(0, 1)) < 0.001);

        Matrix<CheckElement, CheckDevice> grad(1, 2);
        grad.SetValue(0, 0, 0.1f);
        grad.SetValue(0, 1, 0.3f);

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));
        auto fb = Evaluate(out_grad.Get<LayerInput>());

        c = Evaluate(SoftmaxGrad(grad, c));
        assert(fabs(fb(0, 0) - c(0, 0)) < 0.001);
        assert(fabs(fb(0, 1) - c(0, 1)) < 0.001);

        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }
    
    void test_softmax_layer2()
    {
        cout << "Test softmax layer case 2 ...\t";
        using RootLayer = MakeBPLayer<SoftmaxLayer, CommonInputMap, CommonGradMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        vector<Matrix<CheckElement, CheckDevice>> op;

        LayerNeutralInvariant(layer);
        for (size_t loop_count = 1; loop_count < 10; ++loop_count)
        {
            auto in = GenMatrix<CheckElement>(1, loop_count, 0.1f, 0.13f);

            auto input = LayerInputCont<RootLayer>().Set<LayerInput>(in);

            auto out = layer.FeedForward(input);
            auto check = Softmax(in);

            auto handle1 = out.Get<LayerOutput>().EvalRegister();
            auto handle2 = check.EvalRegister();
            EvalPlan<CheckDevice>::Eval();

            auto res = handle1.Data();
            auto c = handle2.Data();

            op.push_back(c);
            for (size_t i = 0; i < loop_count; ++i)
            {
                assert(fabs(res(0, i) - c(0, i)) < 0.0001);
            }
        }

        for (size_t loop_count = 9; loop_count >= 1; --loop_count)
        {
            auto grad = GenMatrix<CheckElement>(1, loop_count, 1.3f, 1.1f);
            auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));
            auto check = SoftmaxGrad(grad, op.back());

            auto handle1 = out_grad.Get<LayerInput>().EvalRegister();
            auto handle2 = check.EvalRegister();
            EvalPlan<CheckDevice>::Eval();

            auto fb = handle1.Data();
            auto c = handle2.Data();
            op.pop_back();

            for (size_t i = 0; i < loop_count; ++i)
            {
                assert(fabs(fb(0, i) - c(0, i)) < 0.0001);
            }
        }

        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }
}

namespace Test::Layer
{
    void test_softmax_layer()
    {
        test_softmax_layer1();
        test_softmax_layer2();
    }
}