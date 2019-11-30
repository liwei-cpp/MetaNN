#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <data_gen.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
    using CommonInputMap = LayerIOMap<LayerKV<LayerInput, Matrix<CheckElement, CheckDevice>>>;

    void test_bias_layer1()
    {
        cout << "Test bias layer case 1 ...\t";
        using RootLayer = MakeInferLayer<BiasLayer, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        static_assert(!RootLayer::IsUpdate, "Test Error");
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");

        RootLayer layer("root", 2, 1);

        // Initialization
        auto filler = MakeInitializer<CheckElement>();
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;

        auto mat = GenMatrix<CheckElement>(2, 1);
        filler.SetParam("root", mat);
        
        layer.Init(filler, loadBuffer);

        auto input = GenMatrix<CheckElement>(2, 1, 0.5f, -0.1f);
        auto bi = LayerInputCont<RootLayer>().Set<LayerInput>(input);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(bi);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(fabs(res(0, 0) - input(0, 0) - mat(0, 0)) < 0.001);
        assert(fabs(res(1, 0) - input(1, 0) - mat(1, 0)) < 0.001);

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<LayerInput>);

        loadBuffer.Clear();
        layer.SaveWeights(loadBuffer);
        auto* w = loadBuffer.TryGet<CategoryTags::Matrix>("root");
        assert(w);
        
        auto wInfo = *w;
        assert(wInfo.Shape() == mat.Shape());
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 1; ++j)
            {
                assert(fabs(wInfo(i, j) - mat(i, j)) < 0.001f);
            }
        }

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
    
    void test_bias_layer2()
    {
        cout << "Test bias layer case 2 ...\t";
        using RootLayer = MakeInferLayer<BiasLayer, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        static_assert(!RootLayer::IsUpdate, "Test Error");
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");

        RootLayer layer("root", 1, 2);

        auto filler = MakeInitializer<CheckElement>();
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;

        auto mat = GenMatrix<CheckElement>(1, 2);
        filler.SetParam("root", mat);
        
        layer.Init(filler, loadBuffer);
    
        auto input = GenMatrix<CheckElement>(1, 2, 0.5f, -0.1f);
        auto bi = LayerInputCont<RootLayer>().Set<LayerInput>(input);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(bi);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(fabs(res(0, 0) - input(0, 0) - mat(0, 0)) < 0.001);
        assert(fabs(res(0, 1) - input(0, 1) - mat(0, 1)) < 0.001);

        auto fbIn = LayerOutputCont<RootLayer>();
        auto out_grad = layer.FeedBackward(fbIn);
        static_assert(decltype(out_grad)::template IsValueEmpty<LayerInput>);

        LayerNeutralInvariant(layer);

        loadBuffer.Clear();
        layer.SaveWeights(loadBuffer);
        assert(loadBuffer.IsParamExist<CategoryTags::Matrix>("root"));

        cout << "done" << endl;
    }
    
    void test_bias_layer3()
    {
        cout << "Test bias layer case 3 ...\t";
        using RootLayer = MakeTrainLayer<BiasLayer, CommonInputMap, PUpdate, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
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

        auto bi = LayerInputCont<RootLayer>().Set<LayerInput>(input);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(bi);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(fabs(res(0, 0) + 0.27f + 0.48f) < 0.001);
        assert(fabs(res(1, 0) + 0.41f + 0.13f) < 0.001);

        Matrix<CheckElement, CheckDevice> g(2, 1);
        g.SetValue(0, 0, -0.0495f);
        g.SetValue(1, 0, -0.0997f);

        auto fbIn = LayerOutputCont<RootLayer>().Set<LayerOutput>(g);
        auto out_grad = layer.FeedBackward(fbIn);

        GradCollector<CheckElement, CheckDevice> grad_collector;
        layer.GradCollect(grad_collector);
        
        auto& gradCont = grad_collector.GetContainer<CategoryTags::Matrix>();
        assert(gradCont.size() == 1);

        auto handle1 = gradCont.front().Weight().EvalRegister();
        auto handle2 = gradCont.front().Grad().EvalRegister();
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
        using RootLayer = MakeTrainLayer<BiasLayer, CommonInputMap, PUpdate, PFeedbackOutput, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
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

        auto bi = LayerInputCont<RootLayer>().Set<LayerInput>(input);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(bi);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(fabs(res(0, 0) + 0.27f + 0.48f) < 0.001);
        assert(fabs(res(1, 0) + 0.41f + 0.13f) < 0.001);

        Matrix<CheckElement, CheckDevice> g(2, 1);
        g.SetValue(0, 0, -0.0495f);
        g.SetValue(1, 0, -0.0997f);

        auto fbIn = LayerOutputCont<RootLayer>().Set<LayerOutput>(g);
        auto out_grad = layer.FeedBackward(fbIn);
        auto fbOut = Evaluate(out_grad.Get<LayerInput>());

        assert(fabs(fbOut(0, 0) + 0.0495f) < 0.001);
        assert(fabs(fbOut(1, 0) + 0.0997f) < 0.001);

        GradCollector<CheckElement, CheckDevice> grad_collector;
        layer.GradCollect(grad_collector);
        auto& gradCont = grad_collector.GetContainer<CategoryTags::Matrix>();
        assert(gradCont.size() == 1);

        auto handle1 = gradCont.front().Weight().EvalRegister();
        auto handle2 = gradCont.front().Grad().EvalRegister();
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
        using RootLayer = MakeTrainLayer<BiasLayer, CommonInputMap, PUpdate, PFeedbackOutput, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
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

        auto bi = LayerInputCont<RootLayer>().Set<LayerInput>(input);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(bi);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(fabs(res(0, 0) + 0.27f + 0.48f) < 0.001);
        assert(fabs(res(1, 0) + 0.41f + 0.13f) < 0.001);

        input = Matrix<CheckElement, CheckDevice>(2, 1);
        input.SetValue(0, 0, 1.27f);
        input.SetValue(1, 0, 2.41f);

        bi = LayerInputCont<RootLayer>().Set<LayerInput>(input);

        out = layer.FeedForward(bi);
        res = Evaluate(out.Get<LayerOutput>());
        assert(fabs(res(0, 0) - 1.27f + 0.48f) < 0.001);
        assert(fabs(res(1, 0) - 2.41f + 0.13f) < 0.001);

        Matrix<CheckElement, CheckDevice> g(2, 1);
        g.SetValue(0, 0, -0.0495f);
        g.SetValue(1, 0, -0.0997f);

        auto fbIn = LayerOutputCont<RootLayer>().Set<LayerOutput>(g);
        auto out_grad = layer.FeedBackward(fbIn);
        auto fbOut = Evaluate(out_grad.Get<LayerInput>());

        assert(fabs(fbOut(0, 0) + 0.0495f) < 0.001);
        assert(fabs(fbOut(1, 0) + 0.0997f) < 0.001);

        g = Matrix<CheckElement, CheckDevice>(2, 1);
        g.SetValue(0, 0, 1.0495f);
        g.SetValue(1, 0, 2.3997f);

        fbIn = LayerOutputCont<RootLayer>().Set<LayerOutput>(g);
        out_grad = layer.FeedBackward(fbIn);
        fbOut = Evaluate(out_grad.Get<LayerInput>());

        assert(fabs(fbOut(0, 0) - 1.0495f) < 0.001);
        assert(fabs(fbOut(1, 0) - 2.3997f) < 0.001);

        GradCollector<CheckElement, CheckDevice> grad_collector;
        layer.GradCollect(grad_collector);
        auto& gradCont = grad_collector.GetContainer<CategoryTags::Matrix>();
        assert(gradCont.size() == 1);

        auto handle1 = gradCont.front().Weight().EvalRegister();
        auto handle2 = gradCont.front().Grad().EvalRegister();
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
        
        struct RootFiller;
        using RootLayer = MakeTrainLayer<BiasLayer, CommonInputMap, PUpdate, PFeedbackOutput, PInitializerIs<RootFiller>, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        static_assert(RootLayer::IsUpdate, "Test Error");
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    
        RootLayer layer("root", 400, 1);
    
        auto initializer = MakeInitializer<CheckElement>(InitializerKV<RootFiller>(ConstantFiller{0}));
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
        struct RootFiller;
        using RootLayer = MakeTrainLayer<BiasLayer, CommonInputMap, PUpdate, PFeedbackOutput, PInitializerIs<RootFiller>, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        static_assert(RootLayer::IsUpdate, "Test Error");
        static_assert(RootLayer::IsFeedbackOutput, "Test Error");

        RootLayer layer("root", 400, 1);
    
        auto initializer = MakeInitializer<CheckElement>(InitializerKV<RootFiller>(ConstantFiller{1.5}));
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

namespace Test::Layer::Compose
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