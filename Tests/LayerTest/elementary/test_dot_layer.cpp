#include <MetaNN/meta_nn.h>
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
    
    void test_dot_layer1()
    {
        cout << "Test dot layer case 1 ...\t";
        using RootLayer = MakeInferLayer<DotLayer>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        auto i1 = GenMatrix<CheckElement>(2, 3, -3.3f, 0.1f);
        auto i2 = GenMatrix<CheckElement>(3, 4, -0.7f, 1.3f);
        auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                .Set<RightOperand>(i2);

        LayerNeutralInvariant(layer);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(res.Shape().RowNum() == 2);
        assert(res.Shape().ColNum() == 4);
        
        auto check = Evaluate(Dot(i1, i2));
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                assert(fabs(res(i, j) - check(i, j)) < 0.001);
            }
        }

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<LeftOperand>);
        static_assert(decltype(out_grad)::template IsValueEmpty<RightOperand>);

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
    
    void test_dot_layer2()
    {
        cout << "Test dot layer case 2 ...\t";
        using RootLayer = MakeTrainLayer<DotLayer, CommonInputMap, PFeedbackOutput>;

        static_assert(RootLayer::IsFeedbackOutput);
        static_assert(!RootLayer::IsUpdate);

        RootLayer layer("root");

        auto i1 = GenMatrix<CheckElement>(2, 3, -3.3f, 0.1f);
        auto i2 = GenMatrix<CheckElement>(3, 4, -0.7f, 1.3f);
        auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                .Set<RightOperand>(i2);

        LayerNeutralInvariant(layer);

        auto out = layer.FeedForward(input);

        auto grad = GenMatrix<CheckElement>(2, 4, 0, 0.1f);
        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));

        auto handle1 = out.Get<LayerOutput>().EvalRegister();
        auto handle2 = out_grad.Get<LeftOperand>().EvalRegister();
        auto handle3 = out_grad.Get<RightOperand>().EvalRegister();
        EvalPlan<CheckDevice>::Eval();

        auto res = handle1.Data();
        assert(res.Shape().RowNum() == 2);
        assert(res.Shape().ColNum() == 4);
        
        auto check = Evaluate(Dot(i1, i2));
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                assert(fabs(res(i, j) - check(i, j)) < 0.001);
            }
        }
        
        auto grad1 = handle2.Data();
        auto check1 = Evaluate(Dot(grad, Transpose(i2)));
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(grad1(i, j) - check1(i, j)) < 0.001);
            }
        }

        auto grad2 = handle3.Data();
        auto check2 = Evaluate(Dot(Transpose(i1), grad));
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                assert(fabs(grad2(i, j) - check2(i, j)) < 0.001);
            }
        }

        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }

    void test_dot_layer3()
    {
        cout << "Test dot layer case 3 ...\t";
        using RootLayer = MakeTrainLayer<DotLayer, CommonInputMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput);
        static_assert(!RootLayer::IsUpdate);

        RootLayer layer("root");

        vector<Matrix<CheckElement, CheckDevice>> op1;
        vector<Matrix<CheckElement, CheckDevice>> op2;
        LayerNeutralInvariant(layer);
        for (size_t loop_count = 1; loop_count < 10; ++loop_count)
        {
            auto i1 = GenMatrix<CheckElement>(2, loop_count, 0, 0.3f);
            auto i2 = GenMatrix<CheckElement>(loop_count, 3, -1, 1.3f);
            op1.push_back(i1);
            op2.push_back(i2);

            auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                    .Set<RightOperand>(i2);

            auto out = layer.FeedForward(input);
            auto res = Evaluate(out.Get<LayerOutput>());
            auto check = Evaluate(Dot(i1, i2));
            assert(res.Shape().RowNum() == 2);
            assert(res.Shape().ColNum() == 3);
            for (size_t i = 0; i < 2; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    assert(fabs(res(i, j) - check(i, j)) < 0.0001);
                }
            }
        }

        for (size_t loop_count = 9; loop_count >= 1; --loop_count)
        {
            auto grad = GenMatrix<CheckElement>(2, 3, 2, 1.1f);
            auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));

            auto handle1 = out_grad.Get<LeftOperand>().EvalRegister();
            auto handle2 = out_grad.Get<RightOperand>().EvalRegister();
            EvalPlan<CheckDevice>::Eval();

            auto g1 = handle1.Data();
            auto g2 = handle2.Data();

            auto i1 = op1.back(); op1.pop_back();
            auto i2 = op2.back(); op2.pop_back();
            auto check1 = Evaluate(Dot(grad, Transpose(i2)));
            auto check2 = Evaluate(Dot(Transpose(i1), grad));
            
            for (size_t i = 0; i < 2; ++i)
            {
                for (size_t j = 0; j < loop_count; ++j)
                {
                    assert(fabs(g1(i, j) - check1(i, j)) < 0.001);
                }
            }
            
            for (size_t i = 0; i < loop_count; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    assert(fabs(g2(i, j) - check2(i, j)) < 0.001);
                }
            }
        }

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }

    void test_dot_layer4()
    {
        cout << "Test dot layer case 4 (dummy grad input)...\t";
        using RootLayer = MakeTrainLayer<DotLayer, CommonInputMap, PFeedbackOutput>;

        static_assert(RootLayer::IsFeedbackOutput);
        static_assert(!RootLayer::IsUpdate);

        RootLayer layer("root");

        auto i1 = GenMatrix<CheckElement>(2, 3, -3.3f, 0.1f);
        auto i2 = GenMatrix<CheckElement>(3, 4, -0.7f, 1.3f);
        auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                .Set<RightOperand>(i2);

        LayerNeutralInvariant(layer);

        auto out = layer.FeedForward(input);

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<LeftOperand>);
        static_assert(decltype(out_grad)::template IsValueEmpty<RightOperand>);

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
}

namespace Test::Layer::Elementary
{
    void test_dot_layer()
    {
        test_dot_layer1();
        test_dot_layer2();
        test_dot_layer3();
        test_dot_layer4();
    }
}
