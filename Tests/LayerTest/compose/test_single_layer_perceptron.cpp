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
    void test_single_layer_perceptron1()
    {
        cout << "Test SLP case 1 (noupdate, sigmoid-activate, with bias)...\t";
        using RootLayer = MakeInferLayer<SingleLayerPerceptron, PActFuncIs<SigmoidLayer>, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
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

        assert(fabs(out(0, 0) - (1 / (1+exp(-0.75)))) < 0.00001);
        assert(fabs(out(0, 1) - (1 / (1+exp(-0.91)))) < 0.00001);
        assert(fabs(out(0, 2) - (1 / (1+exp(-1.07)))) < 0.00001);
        
        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<LayerInput>);
        
        params.Clear();
        layer.SaveWeights(params);
        assert(params.IsParamExist<CategoryTags::Matrix>("root/weight"));
        assert(params.IsParamExist<CategoryTags::Matrix>("root/bias"));

        cout << "done" << endl;
    }
    
    void test_single_layer_perceptron2()
    {
        cout << "Test SLP case 2 (noupdate, sigmoid-activate, without bias)...\t";
        using RootLayer = MakeInferLayer<SingleLayerPerceptron,
                                         PActFuncIs<SigmoidLayer>, PBiasNotInvolved, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);
        
        RootLayer layer("root",
                        Shape<CategoryTags::Matrix>(2, 3));

        Matrix<CheckElement, CheckDevice> w1(2, 3);
        w1.SetValue(0, 0, 0.1f);  w1.SetValue(1, 0, 0.2f);
        w1.SetValue(0, 1, 0.3f);  w1.SetValue(1, 1, 0.4f);
        w1.SetValue(0, 2, 0.5f);  w1.SetValue(1, 2, 0.6f);

        auto initializer = MakeInitializer<CheckElement>();
        initializer.SetParam("root/weight", w1);
        LoadBuffer<CheckElement, CheckDevice> params;
        layer.Init(initializer, params);

        Matrix<CheckElement, CheckDevice> i(1, 2);
        i.SetValue(0, 0, 0.1f); i.SetValue(0, 1, 0.2f);

        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(i);
        auto out = Evaluate(layer.FeedForward(input).Get<LayerOutput>());

        assert(fabs(out(0, 0) - (1 / (1+exp(-0.05)))) < 0.00001);
        assert(fabs(out(0, 1) - (1 / (1+exp(-0.11)))) < 0.00001);
        assert(fabs(out(0, 2) - (1 / (1+exp(-0.17)))) < 0.00001);

        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<LayerInput>);
        
        params.Clear();
        layer.SaveWeights(params);
        assert(params.IsParamExist<CategoryTags::Matrix>("root/weight"));

        cout << "done" << endl;
    }
    
    void test_single_layer_perceptron3()
    {
        cout << "Test SLP case 3 (update, sigmoid-activate, with bias)...\t";
        using RootLayer = MakeTrainLayer<SingleLayerPerceptron, CommonInputMap, PUpdate, PActFuncIs<SigmoidLayer>, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
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

        auto initializer = MakeInitializer<CheckElement>();
        initializer.SetParam("root/weight", w1);
        initializer.SetParam("root/bias", b1);
        LoadBuffer<CheckElement, CheckDevice> params;
        layer.Init(initializer, params);

        Matrix<CheckElement, CheckDevice> i(1, 2);
        i.SetValue(0, 0, 0.1f); i.SetValue(0, 1, 0.2f);

        auto input = LayerInputCont<RootLayer>().Set<LayerInput>(i);
        auto out = Evaluate(layer.FeedForward(input).Get<LayerOutput>());

        assert(fabs(out(0, 0) - (1 / (1+exp(-0.75)))) < 0.00001);
        assert(fabs(out(0, 1) - (1 / (1+exp(-0.91)))) < 0.00001);
        assert(fabs(out(0, 2) - (1 / (1+exp(-1.07)))) < 0.00001);
        
        Matrix<CheckElement, CheckDevice> grad(1, 3);
        grad.SetValue(0, 0, 0.19);
        grad.SetValue(0, 1, 0.23);
        grad.SetValue(0, 2, -0.15);

        auto fbIn = LayerOutputCont<RootLayer>().Set<LayerOutput>(grad);
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

                assert(fabs(info(0, 0) - 0.0414 * 0.1) < 0.00001);
                assert(fabs(info(1, 0) - 0.0414 * 0.2) < 0.00001);
                assert(fabs(info(0, 1) - 0.04706511 * 0.1) < 0.00001);
                assert(fabs(info(1, 1) - 0.04706511 * 0.2) < 0.00001);
                assert(fabs(info(0, 2) + 0.02852585 * 0.1) < 0.00001);
                assert(fabs(info(1, 2) + 0.02852585 * 0.2) < 0.00001);
            }
            else if (w.Shape() == b1.Shape())
            {
                bias_update_valid = true;
                assert(fabs(info(0, 0) - 0.0414) < 0.00001);
                assert(fabs(info(0, 1) - 0.04706511) < 0.00001);
                assert(fabs(info(0, 2) + 0.02852585) < 0.00001);
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
    
    void test_single_layer_perceptron4()
    {
        cout << "Test SLP case 4 (noupdate, tanh-activate, with bias)...\t";
        using RootLayer = MakeInferLayer<SingleLayerPerceptron, PActFuncIs<TanhLayer>, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
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

        assert(fabs(out(0, 0) - tanh(0.75)) < 0.00001);
        assert(fabs(out(0, 1) - tanh(0.91)) < 0.00001);
        assert(fabs(out(0, 2) - tanh(1.07)) < 0.00001);
        
        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>());
        static_assert(decltype(out_grad)::template IsValueEmpty<LayerInput>);
        
        params.Clear();
        layer.SaveWeights(params);
        assert(params.IsParamExist<CategoryTags::Matrix>("root/weight"));
        assert(params.IsParamExist<CategoryTags::Matrix>("root/bias"));

        cout << "done" << endl;
    }
}

namespace Test::Layer::Compose
{
    void test_single_layer_perceptron()
    {
        test_single_layer_perceptron1();
        test_single_layer_perceptron2();
        test_single_layer_perceptron3();
        test_single_layer_perceptron4();
    }
}