#include <MetaNN/meta_nn2.h>
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
        
        using RootLayer = MakeLayer<ParamSourceLayer, NullParameter>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");
        
        auto filler = MakeInitializer<CheckElement>();
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        
        auto mat = GenMatrix<CheckElement>(10, 3);
        filler.SetParam("root", mat);
        
        RootLayer layer("root", 10, 3);
        layer.Init(filler, loadBuffer);
        
        LoadBuffer<CheckElement, CheckDevice> weightSaver;
        layer.SaveWeights(weightSaver);
        auto* w = weightSaver.TryGet<CategoryTags::Matrix>("root");
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

        using RootLayer = MakeLayer<ParamSourceLayer, NullParameter, PInitializerIs<RootFiller>>;
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
        auto* w = weightSaver.TryGet<CategoryTags::Matrix>("root");
        assert(w);
        
        auto wInfo = *w;
        assert(wInfo.Shape().RowNum() == 10);
        assert(wInfo.Shape().ColNum() == 3);
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
        
        using RootLayer = MakeLayer<ParamSourceLayer, NullParameter>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");
        
        auto filler = MakeInitializer<CheckElement>();
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        
        auto mat = GenMatrix<CheckElement>(10, 3);
        loadBuffer.Set("root", mat);

        RootLayer layer("root", 10, 3);
        layer.Init(filler, loadBuffer);
        
        LoadBuffer<CheckElement, CheckDevice> weightSaver;
        layer.SaveWeights(weightSaver);
        auto* w = weightSaver.TryGet<CategoryTags::Matrix>("root");
        assert(w);
        assert(*w == mat);

        cout << "done" << endl;
    }

    void test_param_source_layer4()
    {
        cout << "Test param source layer case 4...\t";
        
        using RootLayer = MakeLayer<ParamSourceLayer, NullParameter>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");
        
        auto filler = MakeInitializer<CheckElement>();
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        
        auto mat = GenMatrix<CheckElement>(10, 3);
        loadBuffer.Set("root", mat);

        RootLayer layer("root", 10, 3);
        layer.Init(filler, loadBuffer);
        
        auto check = layer.FeedForward(NullParameter{});
        auto w = check.template Get<LayerOutput>();
        
        assert(w == mat);

        cout << "done" << endl;
    }

    using CommonGradMap = LayerIOMap<LayerKV<LayerOutput, Matrix<CheckElement, CheckDevice>>>;
    void test_param_source_layer5()
    {
        cout << "Test param source layer case 5...\t";
        
        using RootLayer = MakeBPLayer<ParamSourceLayer, NullParameter, CommonGradMap>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");
        
        auto filler = MakeInitializer<CheckElement>();
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        
        auto mat = GenMatrix<CheckElement>(10, 3);
        loadBuffer.Set("root", mat);

        RootLayer layer("root", 10, 3);
        layer.Init(filler, loadBuffer);
        
        auto fpRes = layer.FeedForward(NullParameter{});
        auto w = fpRes.template Get<LayerOutput>();
        assert(w == mat);
        
        auto grad = GenMatrix<CheckElement>(10, 3);
        
        // Note: although we feed into the BP leayer, but since no update is set, 
        //       the feedbackward has no effect.
        layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));
        layer.NeutralInvariant();
        cout << "done" << endl;
    }
    
    void test_param_source_layer6()
    {
        cout << "Test param source layer case 6...\t";
        
        using RootLayer = MakeBPLayer<ParamSourceLayer, NullParameter, CommonGradMap, PUpdate>;
        static_assert(!RootLayer::IsFeedbackOutput);
        static_assert(RootLayer::IsUpdate);
        
        auto filler = MakeInitializer<CheckElement>();
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        
        auto mat = GenMatrix<CheckElement>(10, 3, 0.1, 3);
        loadBuffer.Set("root", mat);

        RootLayer layer("root", 10, 3);
        layer.Init(filler, loadBuffer);
        
        auto fpRes = layer.FeedForward(NullParameter{});
        auto w = fpRes.template Get<LayerOutput>();
        assert(w == mat);
        
        auto grad = GenMatrix<CheckElement>(10, 3);
        
        layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));
        
        GradCollector<CheckElement, CheckDevice> grad_collector;
        layer.GradCollect(grad_collector);
        
        auto& gradCont = grad_collector.GetContainer<CategoryTags::Matrix>();
        assert(gradCont.size() == 1);

        auto handle1 = gradCont.front().Weight().EvalRegister();
        auto handle2 = gradCont.front().Grad(1).EvalRegister();
        EvalPlan<DeviceTags::CPU>::Eval();
        
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
}

namespace Test::Layer::SrcRec
{
    void test_param_source_layer()
    {
        test_param_source_layer1();
        test_param_source_layer2();
        test_param_source_layer3();
        test_param_source_layer4();
        test_param_source_layer5();
        test_param_source_layer6();
    }
}