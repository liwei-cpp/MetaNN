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

        auto i1 = GenTensor<CheckElement>(-3.3f, 0.1f, 2, 3);
        auto i2 = GenTensor<CheckElement>(-0.7f, 1.3f, 3, 4);
        auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                .Set<RightOperand>(i2);

        LayerNeutralInvariant(layer);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(res.Shape() == Shape(2, 4));
        
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

        auto i1 = GenTensor<CheckElement>(-3.3f, 0.1f, 2, 3);
        auto i2 = GenTensor<CheckElement>(-0.7f, 1.3f, 3, 4);
        auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                .Set<RightOperand>(i2);

        LayerNeutralInvariant(layer);

        auto out = layer.FeedForward(input);

        auto grad = GenTensor<CheckElement>(0, 0.1f, 2, 4);
        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));

        auto handle1 = out.Get<LayerOutput>().EvalRegister();
        auto handle2 = out_grad.Get<LeftOperand>().EvalRegister();
        auto handle3 = out_grad.Get<RightOperand>().EvalRegister();
        EvalPlan<CheckDevice>::Inst().Eval();

        auto res = handle1.Data();
        assert(res.Shape() == Shape(2, 4));
        
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
            auto i1 = GenTensor<CheckElement>(0, 0.3f, 2, loop_count);
            auto i2 = GenTensor<CheckElement>(-1, 1.3f, loop_count, 3);
            op1.push_back(i1);
            op2.push_back(i2);

            auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                    .Set<RightOperand>(i2);

            auto out = layer.FeedForward(input);
            auto res = Evaluate(out.Get<LayerOutput>());
            auto check = Evaluate(Dot(i1, i2));
            assert(res.Shape() == Shape(2, 3));
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
            auto grad = GenTensor<CheckElement>(2, 1.1f, 2, 3);
            auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));

            auto handle1 = out_grad.Get<LeftOperand>().EvalRegister();
            auto handle2 = out_grad.Get<RightOperand>().EvalRegister();
            EvalPlan<CheckDevice>::Inst().Eval();

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

        auto i1 = GenTensor<CheckElement>(-3.3f, 0.1f, 2, 3);
        auto i2 = GenTensor<CheckElement>(-0.7f, 1.3f, 3, 4);
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
    

    void test_dot_layer5()
    {
        cout << "Test dot layer case 5 (multiple dimensions dot)...\t";
        using RootLayer = MakeInferLayer<DotLayer, PModifyDimNumIs<2>>;

        static_assert(!RootLayer::IsFeedbackOutput);
        static_assert(!RootLayer::IsUpdate);

        RootLayer layer("root");

        auto i1 = GenTensor<CheckElement>(-3.3f, 0.1f, 3, 7, 5);
        auto i2 = GenTensor<CheckElement>(-0.7f, 1.3f, 7, 5, 2, 4);
        auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                .Set<RightOperand>(i2);

        LayerNeutralInvariant(layer);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(res.Shape() == Shape(3, 2, 4));
        auto check = Evaluate(Dot<PolicyContainer<PModifyDimNumIs<2>>>(i1, i2));
        
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                for (size_t k = 0; k < 4; ++k)
                {
                    assert(fabs(res(i, j, k) - check(i, j, k)) < 0.001f);
                }
            }
        }

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }

    using MultiDimInputMap = LayerIOMap<LayerKV<LeftOperand, Tensor<CheckElement, CheckDevice, 3>>,
                                        LayerKV<RightOperand, Tensor<CheckElement, CheckDevice, 4>>>;
    void test_dot_layer6()
    {
        cout << "Test dot layer case 6 (multiple dimensions dot with feedback)...\t";
        using RootLayer = MakeTrainLayer<DotLayer, MultiDimInputMap, PFeedbackOutput, PModifyDimNumIs<2>>;

        static_assert(RootLayer::IsFeedbackOutput);
        static_assert(!RootLayer::IsUpdate);

        RootLayer layer("root");

        auto i1 = GenTensor<CheckElement>(-3.3f, 0.1f, 3, 7, 5);
        auto i2 = GenTensor<CheckElement>(-0.7f, 1.3f, 7, 5, 2, 4);
        auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                .Set<RightOperand>(i2);

        LayerNeutralInvariant(layer);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(res.Shape() == Shape(3, 2, 4));
        auto check = Evaluate(Dot<PolicyContainer<PModifyDimNumIs<2>>>(i1, i2));
        
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                for (size_t k = 0; k < 4; ++k)
                {
                    assert(fabs(res(i, j, k) - check(i, j, k)) < 0.001f);
                }
            }
        }

        auto grad = GenTensor<CheckElement>(0, 0.1f, 3, 2, 4);
        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));
        
        auto out_grad1 = Evaluate(out_grad.Get<LeftOperand>());
        auto grad1 = Evaluate(Dot<PolicyContainer<PModifyDimNumIs<2>>>(grad, Permute<PolicyContainer<PDimArrayIs<2, 3, 0, 1>>>(i2)));
        assert(grad1.Shape() == Shape(3, 7, 5));
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 7; ++j)
            {
                for (size_t k = 0; k < 5; ++k)
                {
                    assert(fabs(grad1(i, j, k) - out_grad1(i, j, k)) < 0.001f);
                }
            }
        }

        auto out_grad2 = Evaluate(out_grad.Get<RightOperand>());
        auto grad2 = Evaluate(Dot<PolicyContainer<PModifyDimNumIs<1>>>(Permute<PolicyContainer<PDimArrayIs<1, 2, 0>>>(i1), grad));
        assert(grad2.Shape() == Shape(7, 5, 2, 4));
        for (size_t i = 0; i < 7; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                for (size_t k = 0; k < 2; ++k)
                {
                    for (size_t l = 0; l < 4; ++l)
                    {
                        assert(fabs(grad2(i, j, k, l) - out_grad2(i, j, k, l)) < 0.001f);
                    }
                }
            }
        }

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
        test_dot_layer5();
        test_dot_layer6();
    }
}
