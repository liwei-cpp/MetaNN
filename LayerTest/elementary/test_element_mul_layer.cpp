#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <data_gen.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
    using CommonInputMap = LayerIOMap<LayerKV<LeftOperand, Matrix<CheckElement, CheckDevice>>,
                                      LayerKV<RightOperand, Matrix<CheckElement, CheckDevice>>
                                     >;
    using CommonGradMap = LayerIOMap<LayerKV<LayerOutput, Matrix<CheckElement, CheckDevice>>>;
    
    void test_element_mul_layer1()
    {
        cout << "Test element mul layer case 1 ...\t";
        using RootLayer = MakeLayer<MultiplyLayer, CommonInputMap>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        Matrix<CheckElement, CheckDevice> i1(2, 3);
        i1.SetValue(0, 0, 0.1f);  i1.SetValue(0, 1, 0.2f); i1.SetValue(0, 2, 0.3f);
        i1.SetValue(1, 0, 0.4f);  i1.SetValue(1, 1, 0.5f); i1.SetValue(1, 2, 0.6f);

        Matrix<CheckElement, CheckDevice> i2(2, 3);
        i2.SetValue(0, 0, 0.2f);  i2.SetValue(0, 1, 0.3f); i2.SetValue(0, 2, 0.4f);
        i2.SetValue(1, 0, 0.5f);  i2.SetValue(1, 1, 0.6f); i2.SetValue(1, 2, 0.7f);

        auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                .Set<RightOperand>(i2);

        LayerNeutralInvariant(layer);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(fabs(res(0, 0) - 0.02f) < 0.001);
        assert(fabs(res(0, 1) - 0.06f) < 0.001);
        assert(fabs(res(0, 2) - 0.12f) < 0.001);
        assert(fabs(res(1, 0) - 0.20f) < 0.001);
        assert(fabs(res(1, 1) - 0.30f) < 0.001);
        assert(fabs(res(1, 2) - 0.42f) < 0.001);

        auto out_grad = layer.FeedBackward(NullParameter{});
        auto fb1 = out_grad.Get<LeftOperand>();
        auto fb2 = out_grad.Get<RightOperand>();
        static_assert(std::is_same<decltype(fb1), NullParameter>::value, "Test error");
        static_assert(std::is_same<decltype(fb2), NullParameter>::value, "Test error");

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
    
    void test_element_mul_layer2()
    {
        cout << "Test element mul layer case 2 ...\t";
        using RootLayer = MakeBPLayer<MultiplyLayer, CommonInputMap, CommonGradMap, PFeedbackOutput>;

        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        Matrix<CheckElement, CheckDevice> i1(2, 3);
        i1.SetValue(0, 0, 0.1f);  i1.SetValue(0, 1, 0.2f); i1.SetValue(0, 2, 0.3f);
        i1.SetValue(1, 0, 0.4f);  i1.SetValue(1, 1, 0.5f); i1.SetValue(1, 2, 0.6f);

        Matrix<CheckElement, CheckDevice> i2(2, 3);
        i2.SetValue(0, 0, 0.2f);  i2.SetValue(0, 1, 0.3f); i2.SetValue(0, 2, 0.4f);
        i2.SetValue(1, 0, 0.5f);  i2.SetValue(1, 1, 0.6f); i2.SetValue(1, 2, 0.7f);

        auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                .Set<RightOperand>(i2);

        LayerNeutralInvariant(layer);

        auto out = layer.FeedForward(input);

        Matrix<CheckElement, CheckDevice> grad(2, 3);
        grad.SetValue(0, 0, 0.3f);  grad.SetValue(0, 1, 0.6f); grad.SetValue(0, 2, 0.9f);
        grad.SetValue(1, 0, 0.4f);  grad.SetValue(1, 1, 0.1f); grad.SetValue(1, 2, 0.7f);
        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));

        auto handle1 = out.Get<LayerOutput>().EvalRegister();
        auto handle2 = out_grad.Get<LeftOperand>().EvalRegister();
        auto handle3 = out_grad.Get<RightOperand>().EvalRegister();
        EvalPlan<CheckDevice>::Eval();

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
        using RootLayer = MakeBPLayer<MultiplyLayer, CommonInputMap, CommonGradMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        vector<Matrix<CheckElement, CheckDevice>> op1;
        vector<Matrix<CheckElement, CheckDevice>> op2;
        LayerNeutralInvariant(layer);
        for (size_t loop_count = 1; loop_count < 10; ++loop_count)
        {
            auto i1 = GenMatrix<CheckElement>(loop_count, 3, 0, 0.3f);
            auto i2 = GenMatrix<CheckElement>(loop_count, 3, -1, 1.3f);
            op1.push_back(i1);
            op2.push_back(i2);

            auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                    .Set<RightOperand>(i2);

            auto out = layer.FeedForward(input);
            auto res = Evaluate(out.Get<LayerOutput>());
            assert(res.Shape().RowNum() == loop_count);
            assert(res.Shape().ColNum() == 3);
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
            auto grad = GenMatrix<CheckElement>(loop_count, 3, 2, 1.1f);
            auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));

            auto handle1 = out_grad.Get<LeftOperand>().EvalRegister();
            auto handle2 = out_grad.Get<RightOperand>().EvalRegister();
            EvalPlan<CheckDevice>::Eval();

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

namespace Test::Layer
{
    void test_element_mul_layer()
    {
        test_element_mul_layer1();
        test_element_mul_layer2();
        test_element_mul_layer3();
    }
}