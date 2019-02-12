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
    
    void test_add_layer1()
    {
        cout << "Test add layer case 1 ...\t";
        using RootLayer = MakeLayer<AddLayer, CommonInputMap>;
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

        NullParameter fbIn;
        auto out_grad = layer.FeedBackward(fbIn);
        auto fb1 = out_grad.Get<LeftOperand>();
        auto fb2 = out_grad.Get<RightOperand>();
        static_assert(std::is_same<decltype(fb1), NullParameter>::value, "Test error");
        static_assert(std::is_same<decltype(fb2), NullParameter>::value, "Test error");
        cout << "done" << endl;
    }

    void test_add_layer2()
    {
        cout << "Test add layer case 2 ...\t";

        using RootLayer = MakeBPLayer<AddLayer, CommonInputMap, CommonGradMap, PFeedbackOutput>;
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
        using RootLayer = MakeBPLayer<AddLayer, CommonInputMap, CommonGradMap, PFeedbackOutput>;
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
}

namespace Test::Layer
{
    void test_add_layer()
    {
        test_add_layer1();
        test_add_layer2();
        test_add_layer3();
    }
}
