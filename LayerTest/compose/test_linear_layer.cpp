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

    void test_linear_layer1()
    {
        cout << "Test linear layer case 1 ...\t";
        using RootLayer = MakeInferLayer<LinearLayer>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);

        RootLayer layer("root",
                        Shape<CategoryTags::Matrix>(2, 3),
                        Shape<CategoryTags::Matrix>(1, 3));
    
        Matrix<CheckElement, CheckDevice> w1(2, 3);
        w1.SetValue(0, 0, 0.1f);  w1.SetValue(1, 0, 0.2f);
        w1.SetValue(0, 1, 0.3f);  w1.SetValue(1, 1, 0.4f);
        w1.SetValue(0, 2, 0.5f);  w1.SetValue(1, 2, 0.6f);

        Matrix<CheckElement, CheckDevice> b1(1, 3);
        b1.SetValue(0, 0, 0.7f);  b1.SetValue(0, 1, 0.8f); b1.SetValue(0, 2, 0.9f);
    
        auto initializer = MakeInitializer<CheckElement>();
        initializer.SetParam("root/weight", w1);
        initializer.SetParam("root/bias", b1);
        LoadBuffer<CheckElement, CheckDevice> params;
        layer.Init(initializer, params);

        Matrix<CheckElement, CheckDevice> i(1, 2);
        i.SetValue(0, 0, 0.1f); i.SetValue(0, 1, 0.2f);

        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(i);
        auto out = Evaluate(layer.FeedForward(input).Get<LayerOutput>());

        assert(fabs(out(0, 0) - 0.75f) < 0.00001);
        assert(fabs(out(0, 1) - 0.91f) < 0.00001);
        assert(fabs(out(0, 2) - 1.07f) < 0.00001);

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<LayerInput>);

        params.Clear();
        layer.SaveWeights(params);
        assert(params.IsParamExist<CategoryTags::Matrix>("root/weight"));
        assert(params.IsParamExist<CategoryTags::Matrix>("root/bias"));

        cout << "done" << endl;
    }
    
    void test_linear_layer2()
    {
        cout << "Test linear layer case 2 ...\t";
        using RootLayer = MakeTrainLayer<LinearLayer, CommonInputMap, PUpdate>;
        static_assert(RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);

        RootLayer layer("root",
                        Shape<CategoryTags::Matrix>(2, 3),
                        Shape<CategoryTags::Matrix>(1, 3));

        Matrix<CheckElement, CheckDevice> w1(2, 3);
        w1.SetValue(0, 0, 0.1f);  w1.SetValue(1, 0, 0.2f);
        w1.SetValue(0, 1, 0.3f);  w1.SetValue(1, 1, 0.4f);
        w1.SetValue(0, 2, 0.5f);  w1.SetValue(1, 2, 0.6f);

        Matrix<CheckElement, CheckDevice> b1(1, 3);
        b1.SetValue(0, 0, 0.7f);  b1.SetValue(0, 1, 0.8f); b1.SetValue(0, 2, 0.9f);

        auto initializer = MakeInitializer<float>();
        initializer.SetParam("root/weight", w1);
        initializer.SetParam("root/bias", b1);
        LoadBuffer<CheckElement, CheckDevice> params;
        layer.Init(initializer, params);

        Matrix<CheckElement, CheckDevice> i(1, 2);
        i.SetValue(0, 0, 0.1f); i.SetValue(0, 1, 0.2f);

        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(i);
        auto out = Evaluate(layer.FeedForward(input).Get<LayerOutput>());

        assert(fabs(out(0, 0) - 0.75f) < 0.00001);
        assert(fabs(out(0, 1) - 0.91f) < 0.00001);
        assert(fabs(out(0, 2) - 1.07f) < 0.00001);

        Matrix<CheckElement, CheckDevice> g(1, 3);
        g.SetValue(0, 0, 0.1f);
        g.SetValue(0, 1, 0.2f);
        g.SetValue(0, 2, 0.3f);
        auto fbIn = LayerOutputCont<RootLayer>().Set<LayerOutput>(g);
        auto out_grad = layer.FeedBackward(fbIn);
        static_assert(decltype(out_grad)::template IsValueEmpty<LayerInput>);

        GradCollector<CheckElement, CheckDevice> grad_collector;
        layer.GradCollect(grad_collector);
        auto& gradCont = grad_collector.GetContainer<CategoryTags::Matrix>();
        assert(gradCont.size() == 2);

        bool weight_update_valid = false;
        bool bias_update_valid = false;

        for (auto& p : gradCont)
        {
            auto w = p.Weight();
            auto info = Evaluate(p.Grad());
            if (w.Shape() == w1.Shape())
            {
                weight_update_valid = true;

                auto tmp = Evaluate(Dot(Transpose(i), g));
                assert(tmp.Shape() == info.Shape());

                for (size_t i = 0; i < tmp.Shape().RowNum(); ++i)
                {
                    for (size_t j = 0; j < tmp.Shape().ColNum(); ++j)
                    {
                        assert(fabs(info(i, j) - tmp(i, j)) < 0.0001f);
                    }
                }
            }
            else if (w.Shape() == b1.Shape())
            {
                bias_update_valid = true;
                for (size_t i = 0; i < info.Shape().RowNum(); ++i)
                {
                    for (size_t j = 0; j < info.Shape().ColNum(); ++j)
                    {
                        assert(fabs(info(i, j) - g(i, j)) < 0.0001f);
                    }
                }
            }
            else
            {
                assert(false);
            }
        }
        assert(bias_update_valid);
        assert(weight_update_valid);

        params.Clear();
        layer.SaveWeights(params);
        assert(params.IsParamExist<CategoryTags::Matrix>("root/weight"));
        assert(params.IsParamExist<CategoryTags::Matrix>("root/bias"));

        cout << "done" << endl;
    }
    
    void test_linear_layer3()
    {
        cout << "Test linear layer case 3 (update bias) ...\t";
        using RootLayer = MakeTrainLayer<LinearLayer, CommonInputMap,
                                         PUpdate,
                                         SubPolicyContainer<WeightParamSublayer, PNoUpdate>>;
        static_assert(RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);

        RootLayer layer("root",
                        Shape<CategoryTags::Matrix>(2, 3),
                        Shape<CategoryTags::Matrix>(1, 3));

        Matrix<CheckElement, CheckDevice> w1(2, 3);
        w1.SetValue(0, 0, 0.1f);  w1.SetValue(1, 0, 0.2f);
        w1.SetValue(0, 1, 0.3f);  w1.SetValue(1, 1, 0.4f);
        w1.SetValue(0, 2, 0.5f);  w1.SetValue(1, 2, 0.6f);

        Matrix<CheckElement, CheckDevice> b1(1, 3);
        b1.SetValue(0, 0, 0.7f);  b1.SetValue(0, 1, 0.8f); b1.SetValue(0, 2, 0.9f);
        
        auto initializer = MakeInitializer<float>();
        initializer.SetParam("root/weight", w1);
        initializer.SetParam("root/bias", b1);
        LoadBuffer<CheckElement, CheckDevice> params;
        layer.Init(initializer, params);

        Matrix<CheckElement, CheckDevice> i(1, 2);
        i.SetValue(0, 0, 0.1f); i.SetValue(0, 1, 0.2f);

        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(i);
        auto out = Evaluate(layer.FeedForward(input).Get<LayerOutput>());

        assert(fabs(out(0, 0) - 0.75f) < 0.00001);
        assert(fabs(out(0, 1) - 0.91f) < 0.00001);
        assert(fabs(out(0, 2) - 1.07f) < 0.00001);

        Matrix<CheckElement, CheckDevice> g(1, 3);
        g.SetValue(0, 0, 0.1f);
        g.SetValue(0, 1, 0.2f);
        g.SetValue(0, 2, 0.3f);
        auto fbIn = LayerOutputCont<RootLayer>().Set<LayerOutput>(g);
        auto out_grad = layer.FeedBackward(fbIn);
        static_assert(is_same_v<decltype(out_grad)::ValueType<LayerInput>, NullParameter>);

        GradCollector<CheckElement, CheckDevice> grad_collector;
        layer.GradCollect(grad_collector);
        auto& gradCont = grad_collector.GetContainer<CategoryTags::Matrix>();
        assert(gradCont.size() == 1);

        auto w = gradCont.begin()->Weight();
        assert(w.Shape() == b1.Shape());
        auto info = Evaluate(gradCont.begin()->Grad());
        for (size_t i = 0; i < info.Shape().RowNum(); ++i)
        {
            for (size_t j = 0; j < info.Shape().ColNum(); ++j)
            {
                assert(fabs(w(i, j) - b1(i, j)) < 0.0001f);
                assert(fabs(info(i, j) - g(i, j)) < 0.0001f);
            }
        }

        params.Clear();
        layer.SaveWeights(params);
        assert(params.IsParamExist<CategoryTags::Matrix>("root/weight"));
        assert(params.IsParamExist<CategoryTags::Matrix>("root/bias"));
        cout << "done" << endl;
    }
    
    void test_linear_layer4()
    {
        cout << "Test linear layer case 4 (update weight)...\t";
        using RootLayer = MakeTrainLayer<LinearLayer, CommonInputMap,
                                         PUpdate,
                                         SubPolicyContainer<BiasParamSublayer, PNoUpdate>>;
        static_assert(RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);

        RootLayer layer("root",
                        Shape<CategoryTags::Matrix>(2, 3),
                        Shape<CategoryTags::Matrix>(1, 3));

        Matrix<CheckElement, CheckDevice> w1(2, 3);
        w1.SetValue(0, 0, 0.1f);  w1.SetValue(1, 0, 0.2f);
        w1.SetValue(0, 1, 0.3f);  w1.SetValue(1, 1, 0.4f);
        w1.SetValue(0, 2, 0.5f);  w1.SetValue(1, 2, 0.6f);

        Matrix<CheckElement, CheckDevice> b1(1, 3);
        b1.SetValue(0, 0, 0.7f);  b1.SetValue(0, 1, 0.8f); b1.SetValue(0, 2, 0.9f);
        
        auto initializer = MakeInitializer<float>();
        initializer.SetParam("root/weight", w1);
        initializer.SetParam("root/bias", b1);
        LoadBuffer<CheckElement, CheckDevice> params;
        layer.Init(initializer, params);

        Matrix<CheckElement, CheckDevice> i(1, 2);
        i.SetValue(0, 0, 0.1f); i.SetValue(0, 1, 0.2f);

        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(i);
        auto out = Evaluate(layer.FeedForward(input).Get<LayerOutput>());

        assert(fabs(out(0, 0) - 0.75f) < 0.00001);
        assert(fabs(out(0, 1) - 0.91f) < 0.00001);
        assert(fabs(out(0, 2) - 1.07f) < 0.00001);

        Matrix<CheckElement, CheckDevice> g(1, 3);
        g.SetValue(0, 0, 0.1f);
        g.SetValue(0, 1, 0.2f);
        g.SetValue(0, 2, 0.3f);
        auto fbIn = LayerOutputCont<RootLayer>().Set<LayerOutput>(g);
        auto out_grad = layer.FeedBackward(fbIn);
        static_assert(is_same_v<decltype(out_grad)::ValueType<LayerInput>, NullParameter>);

        GradCollector<CheckElement, CheckDevice> grad_collector;
        layer.GradCollect(grad_collector);
        auto& gradCont = grad_collector.GetContainer<CategoryTags::Matrix>();
        assert(gradCont.size() == 1);
        
        auto w = gradCont.begin()->Weight();
        assert(w.Shape() == w1.Shape());
        
        auto info = Evaluate(gradCont.begin()->Grad());
        auto tmp = Evaluate(Dot(Transpose(i), g));
        for (size_t i = 0; i < info.Shape().RowNum(); ++i)
        {
            for (size_t j = 0; j < info.Shape().ColNum(); ++j)
            {
                assert(fabs(info(i, j) - tmp(i, j)) < 0.0001f);
            }
        }
        
        params.Clear();
        layer.SaveWeights(params);
        assert(params.IsParamExist<CategoryTags::Matrix>("root/weight"));
        assert(params.IsParamExist<CategoryTags::Matrix>("root/bias"));
        cout << "done" << endl;
    }
}

namespace Test::Layer::Compose
{
    void test_linear_layer()
    {
        test_linear_layer1();
        test_linear_layer2();
        test_linear_layer3();
        test_linear_layer4();
    }
}