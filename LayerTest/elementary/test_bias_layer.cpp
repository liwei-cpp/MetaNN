#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <data_gen.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
    using CommonInputMap = LayerIOMap<LayerKV<LayerIO, Matrix<CheckElement, CheckDevice>>>;
    
    void test_bias_layer1()
    {
        cout << "Test bias layer case 1 ...\t";
        using RootLayer = MakeLayer<BiasLayer, CommonInputMap>;
        static_assert(!RootLayer::IsUpdate, "Test Error");
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");

        RootLayer layer("root", 2, 1);
    
        auto initializer = MakeInitializer<CheckElement>();
        auto weight = GenMatrix<CheckElement>(2, 1, 1, 0.1f);
        initializer.SetParam("root", weight);
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        layer.Init(initializer, loadBuffer);

        auto input = GenMatrix<CheckElement>(2, 1, 0.5f, -0.1f);
        auto bi = LayerIO::Create().Set<LayerIO>(input);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(bi);
        auto res = Evaluate(out.Get<LayerIO>());
        assert(fabs(res(0, 0) - input(0, 0) - weight(0, 0)) < 0.001);
        assert(fabs(res(1, 0) - input(1, 0) - weight(1, 0)) < 0.001);

        auto fbIn = LayerIO::Create();
        auto out_grad = layer.FeedBackward(fbIn);
        auto fbOut = out_grad.Get<LayerIO>();
        static_assert(is_same<decltype(fbOut), NullParameter>::value, "Test error");

        loadBuffer.Clear();
        layer.SaveWeights(loadBuffer);
        assert(loadBuffer.IsParamExist<CategoryTags::Matrix>("root"));

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
    
    void test_bias_layer2()
    {
        cout << "Test bias layer case 2 ...\t";
        using RootLayer = MakeLayer<BiasLayer, CommonInputMap>;
        static_assert(!RootLayer::IsUpdate, "Test Error");
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");

        RootLayer layer("root", 1, 2);
    
        auto initializer = MakeInitializer<CheckElement>();
        auto weight = GenMatrix<CheckElement>(1, 2, 1, 0.1f);
        initializer.SetParam("root", weight);
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        layer.Init(initializer, loadBuffer);
    
        auto input = GenMatrix<CheckElement>(1, 2, 0.5f, -0.1f);
        auto bi = LayerIO::Create().Set<LayerIO>(input);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(bi);
        auto res = Evaluate(out.Get<LayerIO>());
        assert(fabs(res(0, 0) - input(0, 0) - weight(0, 0)) < 0.001);
        assert(fabs(res(0, 1) - input(0, 1) - weight(0, 1)) < 0.001);

        auto fbIn = LayerIO::Create();
        auto out_grad = layer.FeedBackward(fbIn);
        auto fbOut = out_grad.Get<LayerIO>();
        static_assert(is_same<decltype(fbOut), NullParameter>::value, "Test error");

        LayerNeutralInvariant(layer);

        loadBuffer.Clear();
        layer.SaveWeights(loadBuffer);
        assert(loadBuffer.IsParamExist<CategoryTags::Matrix>("root"));

        cout << "done" << endl;
    }
    
    void test_bias_layer3()
    {
        cout << "Test bias layer case 3 ...\t";
        using RootLayer = MakeBPLayer<BiasLayer, CommonInputMap, CommonInputMap, PUpdate>;
        static_assert(RootLayer::IsUpdate, "Test Error");
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");

        RootLayer layer("root", 2, 1);

        Matrix<CheckElement, CheckDevice> w(2, 1);
        w.SetValue(0, 0, -0.48f);
        w.SetValue(1, 0, -0.13f);

        auto initializer = MakeInitializer<CheckElement>();
        initializer.SetParam("root", w);
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        layer.Init(initializer, loadBuffer);

        Matrix<CheckElement, CheckDevice> input(2, 1);
        input.SetValue(0, 0, -0.27f);
        input.SetValue(1, 0, -0.41f);

        auto bi = LayerIO::Create().Set<LayerIO>(input);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(bi);
        auto res = Evaluate(out.Get<LayerIO>());
        assert(fabs(res(0, 0) + 0.27f + 0.48f) < 0.001);
        assert(fabs(res(1, 0) + 0.41f + 0.13f) < 0.001);

        Matrix<CheckElement, CheckDevice> g(2, 1);
        g.SetValue(0, 0, -0.0495f);
        g.SetValue(1, 0, -0.0997f);

        auto fbIn = LayerIO::Create().Set<LayerIO>(g);
        auto out_grad = layer.FeedBackward(fbIn);

        GradCollector<CheckElement, CheckDevice> grad_collector;
        layer.GradCollect(grad_collector);
        
        auto& gradCont = grad_collector.GetContainer<CategoryTags::Matrix>();
        assert(gradCont.size() == 1);

        auto handle1 = gradCont.front().Weight().EvalRegister();
        auto handle2 = gradCont.front().Grad(1).EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto w1 = handle1.Data();
        auto g1 = handle2.Data();

        assert(fabs(w1(0, 0) + 0.48f) < 0.001);
        assert(fabs(w1(1, 0) + 0.13f) < 0.001);
        assert(fabs(g1(0, 0) + 0.0495f) < 0.001);
        assert(fabs(g1(1, 0) + 0.0997f) < 0.001);
        LayerNeutralInvariant(layer);

        loadBuffer.Clear();
        layer.SaveWeights(loadBuffer);
        assert(loadBuffer.IsParamExist<CategoryTags::Matrix>("root"));

        cout << "done" << endl;
    }
    
    void test_bias_layer4()
    {
        cout << "Test bias layer case 4 ...\t";
        using RootLayer = MakeBPLayer<BiasLayer, CommonInputMap, CommonInputMap, PUpdate, PFeedbackOutput>;
        static_assert(RootLayer::IsUpdate, "Test Error");
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");

        RootLayer layer("root", 2, 1);
    
        Matrix<CheckElement, CheckDevice> w(2, 1);
        w.SetValue(0, 0, -0.48f);
        w.SetValue(1, 0, -0.13f);

        auto initializer = MakeInitializer<CheckElement>();
        initializer.SetParam("root", w);
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        layer.Init(initializer, loadBuffer);

        Matrix<CheckElement, CheckDevice> input(2, 1);
        input.SetValue(0, 0, -0.27f);
        input.SetValue(1, 0, -0.41f);

        auto bi = LayerIO::Create().Set<LayerIO>(input);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(bi);
        auto res = Evaluate(out.Get<LayerIO>());
        assert(fabs(res(0, 0) + 0.27f + 0.48f) < 0.001);
        assert(fabs(res(1, 0) + 0.41f + 0.13f) < 0.001);

        Matrix<CheckElement, CheckDevice> g(2, 1);
        g.SetValue(0, 0, -0.0495f);
        g.SetValue(1, 0, -0.0997f);

        auto fbIn = LayerIO::Create().Set<LayerIO>(g);
        auto out_grad = layer.FeedBackward(fbIn);
        auto fbOut = Evaluate(out_grad.Get<LayerIO>());

        assert(fabs(fbOut(0, 0) + 0.0495f) < 0.001);
        assert(fabs(fbOut(1, 0) + 0.0997f) < 0.001);

        GradCollector<CheckElement, CheckDevice> grad_collector;
        layer.GradCollect(grad_collector);
        auto& gradCont = grad_collector.GetContainer<CategoryTags::Matrix>();
        assert(gradCont.size() == 1);

        auto handle1 = gradCont.front().Weight().EvalRegister();
        auto handle2 = gradCont.front().Grad(1).EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto w1 = handle1.Data();
        auto g1 = handle2.Data();

        assert(fabs(w1(0, 0) + 0.48f) < 0.001);
        assert(fabs(w1(1, 0) + 0.13f) < 0.001);
        assert(fabs(g1(0, 0) + 0.0495f) < 0.001);
        assert(fabs(g1(1, 0) + 0.0997f) < 0.001);
        LayerNeutralInvariant(layer);

        loadBuffer.Clear();
        layer.SaveWeights(loadBuffer);
        assert(loadBuffer.IsParamExist<CategoryTags::Matrix>("root"));

        cout << "done" << endl;
    }
    
    void test_bias_layer5()
    {
        cout << "Test bias layer case 5 ...\t";
        using RootLayer = MakeBPLayer<BiasLayer, CommonInputMap, CommonInputMap, PUpdate, PFeedbackOutput>;
        static_assert(RootLayer::IsUpdate, "Test Error");
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");

        RootLayer layer("root", 2, 1);
    
        Matrix<CheckElement, CheckDevice> w(2, 1);
        w.SetValue(0, 0, -0.48f);
        w.SetValue(1, 0, -0.13f);
        
        auto initializer = MakeInitializer<float>();
        initializer.SetParam("root", w);
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        layer.Init(initializer, loadBuffer);

        Matrix<CheckElement, CheckDevice> input(2, 1);
        input.SetValue(0, 0, -0.27f);
        input.SetValue(1, 0, -0.41f);

        auto bi = LayerIO::Create().Set<LayerIO>(input);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(bi);
        auto res = Evaluate(out.Get<LayerIO>());
        assert(fabs(res(0, 0) + 0.27f + 0.48f) < 0.001);
        assert(fabs(res(1, 0) + 0.41f + 0.13f) < 0.001);

        input = Matrix<CheckElement, CheckDevice>(2, 1);
        input.SetValue(0, 0, 1.27f);
        input.SetValue(1, 0, 2.41f);

        bi = LayerIO::Create().Set<LayerIO>(input);

        out = layer.FeedForward(bi);
        res = Evaluate(out.Get<LayerIO>());
        assert(fabs(res(0, 0) - 1.27f + 0.48f) < 0.001);
        assert(fabs(res(1, 0) - 2.41f + 0.13f) < 0.001);

        Matrix<CheckElement, CheckDevice> g(2, 1);
        g.SetValue(0, 0, -0.0495f);
        g.SetValue(1, 0, -0.0997f);

        auto fbIn = LayerIO::Create().Set<LayerIO>(g);
        auto out_grad = layer.FeedBackward(fbIn);
        auto fbOut = Evaluate(out_grad.Get<LayerIO>());

        assert(fabs(fbOut(0, 0) + 0.0495f) < 0.001);
        assert(fabs(fbOut(1, 0) + 0.0997f) < 0.001);

        g = Matrix<CheckElement, CheckDevice>(2, 1);
        g.SetValue(0, 0, 1.0495f);
        g.SetValue(1, 0, 2.3997f);

        fbIn = LayerIO::Create().Set<LayerIO>(g);
        out_grad = layer.FeedBackward(fbIn);
        fbOut = Evaluate(out_grad.Get<LayerIO>());

        assert(fabs(fbOut(0, 0) - 1.0495f) < 0.001);
        assert(fabs(fbOut(1, 0) - 2.3997f) < 0.001);

        GradCollector<CheckElement, CheckDevice> grad_collector;
        layer.GradCollect(grad_collector);
        auto& gradCont = grad_collector.GetContainer<CategoryTags::Matrix>();
        assert(gradCont.size() == 1);

        auto handle1 = gradCont.front().Weight().EvalRegister();
        auto handle2 = gradCont.front().Grad(1).EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();

        auto w1 = handle1.Data();
        auto g1 = handle2.Data();

        assert(fabs(w1(0, 0) + 0.48f) < 0.001);
        assert(fabs(w1(1, 0) + 0.13f) < 0.001);
        assert(fabs(g1(0, 0) + 0.0495f - 1.0495f) < 0.001);
        assert(fabs(g1(1, 0) + 0.0997f - 2.3997f) < 0.001);
        LayerNeutralInvariant(layer);

        loadBuffer.Clear();
        layer.SaveWeights(loadBuffer);
        assert(loadBuffer.IsParamExist<CategoryTags::Matrix>("root"));
        cout << "done" << endl;
    }
    
    void test_bias_layer6()
    {
        cout << "Test bias layer case 6 ...\t";
        using RootLayer = MakeBPLayer<BiasLayer, CommonInputMap, CommonInputMap, PUpdate, PFeedbackOutput>;
        static_assert(RootLayer::IsUpdate, "Test Error");
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    
        RootLayer layer("root", 400, 1);
    
        auto initializer = MakeInitializer<CheckElement, PInitializerIs<struct ConstantTag>>()
                                .SetFiller<ConstantTag>(ConstantFiller{0});
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        layer.Init(initializer, loadBuffer);
        assert(loadBuffer.IsParamExist<CategoryTags::Matrix>("root"));
    
        auto val = loadBuffer.TryGet<CategoryTags::Matrix>("root");
        assert(val);
    
        for (size_t i = 0; i < val->Shape().RowNum(); ++i)
        {
            for (size_t j = 0; j < val->Shape().ColNum(); ++j)
            {
                assert(fabs((*val)(i, j)) < 0.0001);
            }
        }
        cout << "done" << endl;
    }
    
    void test_bias_layer7()
    {
        cout << "Test bias layer case 7 ...\t";
        using RootLayer = MakeBPLayer<BiasLayer, CommonInputMap, CommonInputMap, PUpdate, PFeedbackOutput>;
        static_assert(RootLayer::IsUpdate, "Test Error");
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");

        RootLayer layer("root", 400, 1);
    
        auto initializer = MakeInitializer<float, PInitializerIs<struct ConstantTag>>()
                                .SetFiller<ConstantTag>(ConstantFiller{1.5});
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        layer.Init(initializer, loadBuffer);
        assert(loadBuffer.IsParamExist<CategoryTags::Matrix>("root"));
    
        auto val = loadBuffer.TryGet<CategoryTags::Matrix>("root");
        assert(val);
    
        for (size_t i = 0; i < val->Shape().RowNum(); ++i)
        {
            for (size_t j = 0; j < val->Shape().ColNum(); ++j)
            {
                assert(fabs((*val)(i, j) - 1.5) < 0.0001);
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Layer
{
    void test_bias_layer()
    {
        test_bias_layer1();
        test_bias_layer2();
        test_bias_layer3();
        test_bias_layer4();
        test_bias_layer5();
        test_bias_layer6();
        test_bias_layer7();
    }
}