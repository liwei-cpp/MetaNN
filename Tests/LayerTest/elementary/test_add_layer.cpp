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
    void test_add_layer1()
    {
        cout << "Test add layer case 1 ...\t";
        using RootLayer = MakeInferLayer<AddLayer>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        auto i1 = GenMatrix<CheckElement>(2, 3, 1, 0.1f);
        auto i2 = GenMatrix<CheckElement>(2, 3, 1.5f, -0.1f);

        auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                .Set<RightOperand>(i2);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(res(i, j) - i1(i, j) - i2(i, j)) < 0.001);
            }
        }

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<LeftOperand>);
        static_assert(decltype(out_grad)::template IsValueEmpty<RightOperand>);
        cout << "done" << endl;
    }

    void test_add_layer2()
    {
        cout << "Test add layer case 2 ...\t";

        using RootLayer = MakeTrainLayer<AddLayer, CommonInputMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");
        auto i1 = GenMatrix<CheckElement>(2, 3, 1, 0.1f);
        auto i2 = GenMatrix<CheckElement>(2, 3, 1.5f, -0.1f);

        auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                .Set<RightOperand>(i2);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(res(i, j) - i1(i, j) - i2(i, j)) < 0.001);
            }
        }

        auto grad = GenMatrix<CheckElement>(2, 3, 0.7f, -0.2f);

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));

        auto handle1 = out_grad.Get<LeftOperand>().EvalRegister();
        auto handle2 = out_grad.Get<RightOperand>().EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto fb1 = handle1.Data();
        auto fb2 = handle2.Data();
        assert(fb1.Shape().RowNum() == fb2.Shape().RowNum());
        assert(fb1.Shape().ColNum() == fb2.Shape().ColNum());
        assert(fb1.Shape().RowNum() == 2);
        assert(fb1.Shape().ColNum() == 3);

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
        using RootLayer = MakeTrainLayer<AddLayer, CommonInputMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        auto i1 = GenMatrix<CheckElement>(2, 3, 1, 0.1f);
        auto i2 = GenMatrix<CheckElement>(2, 3, 1.5f, -0.1f);

        auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                .Set<RightOperand>(i2);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(res(i, j) - i1(i, j) - i2(i, j)) < 0.001);
            }
        }

        auto i3 = GenMatrix<CheckElement>(2, 3, 1.3, -0.1f);
        auto i4 = GenMatrix<CheckElement>(2, 3, 2.5f, -0.7f);

        input = LayerInputCont<RootLayer>().Set<LeftOperand>(i3).Set<RightOperand>(i4);

        out = layer.FeedForward(input);
        res = Evaluate(out.Get<LayerOutput>());
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(res(i, j) - i3(i, j) - i4(i, j)) < 0.001);
            }
        }

        auto grad = GenMatrix<CheckElement>(2, 3, 0.7f, -0.2f);

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));

        auto handle1 = out_grad.Get<LeftOperand>().EvalRegister();
        auto handle2 = out_grad.Get<RightOperand>().EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto fb1 = handle1.Data();
        auto fb2 = handle2.Data();
        assert(fb1.Shape().RowNum() == fb2.Shape().RowNum());
        assert(fb1.Shape().ColNum() == fb2.Shape().ColNum());
        assert(fb1.Shape().RowNum() == 2);
        assert(fb1.Shape().ColNum() == 3);

        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fb1(i, j) == grad(i, j));
                assert(fb2(i, j) == grad(i, j));
            }
        }

        grad = GenMatrix<CheckElement>(2, 3, -0.7f, 0.2f);

        out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));

        handle1 = out_grad.Get<LeftOperand>().EvalRegister();
        handle2 = out_grad.Get<RightOperand>().EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        fb1 = handle1.Data();
        fb2 = handle2.Data();

        assert(fb1.Shape().RowNum() == fb2.Shape().RowNum());
        assert(fb1.Shape().ColNum() == fb2.Shape().ColNum());
        assert(fb1.Shape().RowNum() == 2);
        assert(fb1.Shape().ColNum() == 3);

        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fb1(i, j) == grad(i, j));
                assert(fb2(i, j) == grad(i, j));
            }
        }
        layer.NeutralInvariant();
        cout << "done" << endl;
    }
    
    void test_add_layer4()
    {
        cout << "Test add layer case 4 (add with number)...\t";
        
        using InputMap = LayerIOMap<LayerKV<LeftOperand, Matrix<CheckElement, CheckDevice>>,
                                    LayerKV<RightOperand, int>
                                   >;

        using RootLayer = MakeTrainLayer<AddLayer, InputMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");
        auto i1 = GenMatrix<CheckElement>(2, 3, 1, 0.1f);

        auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                .Set<RightOperand>(3.3f);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                // Note: since RightOperand should be int, the 3.3f should be translated into 3.
                assert(fabs(res(i, j) - i1(i, j) - 3) < 0.001);
            }
        }

        auto grad = GenMatrix<CheckElement>(2, 3, 0.7f, -0.2f);

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));

        auto handle1 = out_grad.Get<LeftOperand>().EvalRegister();
        static_assert(std::is_same_v<RemConstRef<decltype(out_grad.Get<RightOperand>())>, NullParameter>);
        EvalPlan<DeviceTags::CPU>::Eval();

        auto fb1 = handle1.Data();
        assert(fb1.Shape().RowNum() == 2);
        assert(fb1.Shape().ColNum() == 3);

        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fb1(i, j) == grad(i, j));
            }
        }
        layer.NeutralInvariant();
        cout << "done" << endl;
    }
    
    void test_add_layer5()
    {
        cout << "Test add layer case 5 (add with number 2)...\t";
        
        using InputMap = LayerIOMap<LayerKV<LeftOperand, int>,
                                    LayerKV<RightOperand, Matrix<CheckElement, CheckDevice>>
                                   >;

        using RootLayer = MakeTrainLayer<AddLayer, InputMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");
        auto i1 = GenMatrix<CheckElement>(2, 3, 1, 0.1f);

        auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(3.3f)
                                                .Set<RightOperand>(i1);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                // Note: since RightOperand should be int, the 3.3f should be translated into 3.
                assert(fabs(res(i, j) - i1(i, j) - 3) < 0.001);
            }
        }

        auto grad = GenMatrix<CheckElement>(2, 3, 0.7f, -0.2f);

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));

        auto handle1 = out_grad.Get<RightOperand>().EvalRegister();
        static_assert(std::is_same_v<RemConstRef<decltype(out_grad.Get<LeftOperand>())>, NullParameter>);
        EvalPlan<DeviceTags::CPU>::Eval();

        auto fb1 = handle1.Data();
        assert(fb1.Shape().RowNum() == 2);
        assert(fb1.Shape().ColNum() == 3);

        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fb1(i, j) == grad(i, j));
            }
        }
        layer.NeutralInvariant();
        cout << "done" << endl;
    }
    
    void test_add_layer6()
    {
        cout << "Test add layer case 6 (dummy grad input)...\t";

        using RootLayer = MakeTrainLayer<AddLayer, CommonInputMap, PFeedbackOutput>;
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");
        auto i1 = GenMatrix<CheckElement>(2, 3, 1, 0.1f);
        auto i2 = GenMatrix<CheckElement>(2, 3, 1.5f, -0.1f);

        auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                .Set<RightOperand>(i2);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(res(i, j) - i1(i, j) - i2(i, j)) < 0.001);
            }
        }

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<LeftOperand>);
        static_assert(decltype(out_grad)::template IsValueEmpty<RightOperand>);
        layer.NeutralInvariant();
        cout << "done" << endl;
    }
}

namespace Test::Layer::Elementary
{
    void test_add_layer()
    {
        test_add_layer1();
        test_add_layer2();
        test_add_layer3();
        test_add_layer4();
        test_add_layer5();
        test_add_layer6();
    }
}
