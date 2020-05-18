#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <data_gen.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_param_source_layer1()
    {
        cout << "Test param source layer case 1 (Init from initializer-data)...\t";
        
        using RootLayer = MakeInferLayer<ParamSourceLayer, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");
        
        auto filler = MakeInitializer<CheckElement>();
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        
        auto mat = GenTensor<CheckElement>(0, 1, 10, 3);
        filler.SetParam("root", mat);
        
        RootLayer layer("root", 10, 3);
        layer.Init(filler, loadBuffer);
        
        LoadBuffer<CheckElement, CheckDevice> weightSaver;
        layer.SaveWeights(weightSaver);
        auto* w = weightSaver.TryGet<CategoryTags::Tensor<2>>("root");
        assert(w);
        
        auto wInfo = *w;
        assert(wInfo.Shape() == mat.Shape());
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(wInfo(i, j) - mat(i, j)) < 0.001f);
            }
        }

        cout << "done" << endl;
    }

    void test_param_source_layer2()
    {
        cout << "Test param source layer case 2 (Init from initializer-filler)...\t";
        
        struct RootFiller;

        using RootLayer = MakeInferLayer<ParamSourceLayer, PInitializerIs<RootFiller>, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");
        
        auto filler
            = MakeInitializer<CheckElement>
                (InitializerKV<RootFiller>(ConstantFiller{3}));
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        
        RootLayer layer("root", 10, 3);
        layer.Init(filler, loadBuffer);
        
        LoadBuffer<CheckElement, CheckDevice> weightSaver;
        layer.SaveWeights(weightSaver);
        auto* w = weightSaver.TryGet<CategoryTags::Tensor<2>>("root");
        assert(w);
        
        auto wInfo = *w;
        assert(wInfo.Shape() == Shape(10, 3));
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(wInfo(i, j) - 3) < 0.001f);
            }
        }

        cout << "done" << endl;
    }

    void test_param_source_layer3()
    {
        cout << "Test param source layer case 3 (Init from load buffer)...\t";
        
        using RootLayer = MakeInferLayer<ParamSourceLayer, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");
        
        auto filler = MakeInitializer<CheckElement>();
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        
        auto mat = GenTensor<CheckElement>(0, 1, 10, 3);
        loadBuffer.Set("root", mat);

        RootLayer layer("root", 10, 3);
        layer.Init(filler, loadBuffer);
        
        LoadBuffer<CheckElement, CheckDevice> weightSaver;
        layer.SaveWeights(weightSaver);
        auto* w = weightSaver.TryGet<CategoryTags::Tensor<2>>("root");
        assert(w);
        assert(*w == mat);

        cout << "done" << endl;
    }

    void test_param_source_layer4()
    {
        cout << "Test param source layer case 4...\t";
        
        using RootLayer = MakeInferLayer<ParamSourceLayer, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");
        
        auto filler = MakeInitializer<CheckElement>();
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        
        auto mat = GenTensor<CheckElement>(0, 1, 10, 3);
        loadBuffer.Set("root", mat);

        RootLayer layer("root", 10, 3);
        layer.Init(filler, loadBuffer);
        
        auto check = layer.FeedForward(LayerInputCont<RootLayer>());
        auto w = check.template Get<LayerOutput>();
        
        assert(w == mat);

        cout << "done" << endl;
    }

    void test_param_source_layer5()
    {
        cout << "Test param source layer case 5...\t";
        
        using RootLayer = MakeTrainLayer<ParamSourceLayer, LayerIOMap<>, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");
        
        auto filler = MakeInitializer<CheckElement>();
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        
        auto mat = GenTensor<CheckElement>(0, 1, 10, 3);
        loadBuffer.Set("root", mat);

        RootLayer layer("root", 10, 3);
        layer.Init(filler, loadBuffer);
        
        auto fpRes = layer.FeedForward(LayerInputCont<RootLayer>());
        auto w = fpRes.template Get<LayerOutput>();
        assert(w == mat);
        
        auto grad = GenTensor<CheckElement>(0, 1, 10, 3);
        
        // Note: although we feed into the BP leayer, but since no update is set, 
        //       the feedbackward has no effect.
        layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));
        layer.NeutralInvariant();
        cout << "done" << endl;
    }

    void test_param_source_layer6()
    {
        cout << "Test param source layer case 6...\t";
        
        using RootLayer = MakeTrainLayer<ParamSourceLayer, LayerIOMap<>, PUpdate, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        static_assert(!RootLayer::IsFeedbackOutput);
        static_assert(RootLayer::IsUpdate);
        
        auto filler = MakeInitializer<CheckElement>();
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        
        auto mat = GenTensor<CheckElement>(0.1, 3, 10, 3);
        loadBuffer.Set("root", mat);

        RootLayer layer("root", 10, 3);
        layer.Init(filler, loadBuffer);
        
        auto fpRes = layer.FeedForward(LayerInputCont<RootLayer>());
        auto w = fpRes.template Get<LayerOutput>();
        assert(w == mat);
        
        auto grad = GenTensor<CheckElement>(0, 1, 10, 3);
        
        layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));
        
        GradCollector<CheckElement, CheckDevice> grad_collector;
        layer.GradCollect(grad_collector);
        
        auto& gradCont = grad_collector.GetContainer<CategoryTags::Tensor<2>>();
        assert(gradCont.size() == 1);

        auto handle1 = gradCont.begin()->second.Weight().EvalRegister();
        auto handle2 = gradCont.begin()->second.Grad().EvalRegister();
        EvalPlan<CheckDevice>::Inst().Eval();
        
        auto res1 = handle1.Data();
        auto resg = handle2.Data();
        assert(res1.Shape() == resg.Shape());
        assert(res1.Shape() == mat.Shape());
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(res1(i, j) - mat(i, j)) < 0.001f);
                assert(fabs(resg(i, j) - grad(i, j)) < 0.001f);
            }
        }
        
        layer.NeutralInvariant();
        cout << "done" << endl;
    }

    void test_param_source_layer7()
    {
        cout << "Test param source layer case 7 (dummy grad input)...\t";
        
        using RootLayer = MakeTrainLayer<ParamSourceLayer, LayerIOMap<>, PUpdate, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        static_assert(!RootLayer::IsFeedbackOutput);
        static_assert(RootLayer::IsUpdate);
        
        auto filler = MakeInitializer<CheckElement>();
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        
        auto mat = GenTensor<CheckElement>(0, 1, 10, 3);
        loadBuffer.Set("root", mat);

        RootLayer layer("root", 10, 3);
        layer.Init(filler, loadBuffer);
        
        auto fpRes = layer.FeedForward(LayerInputCont<RootLayer>());
        auto w = fpRes.template Get<LayerOutput>();
        assert(w == mat);
        
        auto grad = GenTensor<CheckElement>(0, 1, 10, 3);
        
        layer.FeedBackward(LayerOutputCont<RootLayer>());
        layer.NeutralInvariant();
        cout << "done" << endl;
    }
    
    void test_param_source_layer8()
    {
        cout << "Test param source layer case 8 (ZeroMatrix as parameter)...\t";
        
        using RootLayer = MakeTrainLayer<ParamSourceLayer, LayerIOMap<>, PUpdate, PParamTypeIs<ZeroTensor<CheckElement, CheckDevice, 2>>>;
        static_assert(!RootLayer::IsFeedbackOutput);
        static_assert(!RootLayer::IsUpdate);
        
        RootLayer layer("root", 10, 3);
        
        auto fpRes = layer.FeedForward(LayerInputCont<RootLayer>());
        auto w = fpRes.template Get<LayerOutput>();
        static_assert(std::is_same_v<decltype(w), ZeroTensor<CheckElement, CheckDevice, 2>>);
        assert(w.Shape() == Shape(10, 3));
        
        auto grad = GenTensor<CheckElement>(0, 1, 10, 3);
        
        layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));
        layer.NeutralInvariant();
        cout << "done" << endl;
    }
}

namespace Test::Layer::Source
{
    void test_param_source_layer()
    {
        test_param_source_layer1();
        test_param_source_layer2();
        test_param_source_layer3();
        test_param_source_layer4();
        test_param_source_layer5();
        test_param_source_layer6();
        test_param_source_layer7();
        test_param_source_layer8();
    }
}