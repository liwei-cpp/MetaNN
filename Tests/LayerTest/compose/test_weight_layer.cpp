#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <data_gen.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
    using CommonInputMap = LayerInMap<LayerKV<LayerInput, Matrix<CheckElement, CheckDevice>>>;
    
    void test_weight_layer1()
    {
        cout << "Test weight layer case 1 ...\t" << flush;
        using RootLayer = MakeInferLayer<WeightLayer, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        static_assert(!RootLayer::IsFeedbackOutput);
        static_assert(!RootLayer::IsUpdate);

        RootLayer layer("root", 1, 2);
    
        Matrix<CheckElement, CheckDevice> w(1, 2);
        w.SetValue(0, 0, -0.27f);
        w.SetValue(0, 1, -0.41f);
    
        auto initializer = MakeInitializer<CheckElement>();
        initializer.SetParam("root/param", w);
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        layer.Init(initializer, loadBuffer);
    
        Matrix<CheckElement, CheckDevice> input(1, 1);
        input.SetValue(0, 0, 1);

        LayerNeutralInvariant(layer);
        auto wi = LayerInputCont<RootLayer>().Set<LayerInput>(input);

        auto out = layer.FeedForward(wi);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(fabs(res(0, 0) + 0.27f) < 0.001);
        assert(fabs(res(0, 1) + 0.41f) < 0.001);

        auto out_grad = layer.FeedBackward(NullParameter{});
        static_assert(decltype(out_grad)::template IsValueEmpty<LayerInput>);

        loadBuffer.Clear();
        layer.SaveWeights(loadBuffer);
        assert(loadBuffer.IsParamExist<CategoryTags::Matrix>("root/param"));

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
    
    void test_weight_layer2()
    {
        cout << "Test weight layer case 2 ...\t" << flush;
        using RootLayer = MakeTrainLayer<WeightLayer, CommonInputMap, PUpdate, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        static_assert(!RootLayer::IsFeedbackOutput);
        static_assert(RootLayer::IsUpdate);

        RootLayer layer("root", 1, 2);
    
        Matrix<CheckElement, CheckDevice> w(1, 2);
        w.SetValue(0, 0, -0.27f);
        w.SetValue(0, 1, -0.41f);
    
        auto initializer = MakeInitializer<CheckElement>();
        initializer.SetParam("root/param", w);
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        layer.Init(initializer, loadBuffer);
    
        Matrix<CheckElement, CheckDevice> input(1, 1);
        input.SetValue(0, 0, 0.1f);

        auto wi = LayerInputCont<RootLayer>().Set<LayerInput>(input);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(wi);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(fabs(res(0, 0) + 0.027f) < 0.001);
        assert(fabs(res(0, 1) + 0.041f) < 0.001);

        Matrix<CheckElement, CheckDevice> g(1, 2);
        g.SetValue(0, 0, -0.0495f);
        g.SetValue(0, 1, -0.0997f);
        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(g));
        static_assert(decltype(out_grad)::template IsValueEmpty<LayerInput>);

        GradCollector<CheckElement, CheckDevice> grad_collector;
        layer.GradCollect(grad_collector);
        auto& gradCont = grad_collector.GetContainer<CategoryTags::Matrix>();
        assert(gradCont.size() == 1);

        auto handle1 = gradCont.begin()->second.Weight().EvalRegister();
        auto handle2 = gradCont.begin()->second.Grad().EvalRegister();
        EvalPlan<CheckDevice>::Inst().Eval();

        auto w1 = handle1.Data();
        auto g1 = handle2.Data();

        assert(fabs(w1(0, 0) + 0.27f) < 0.001);
        assert(fabs(w1(0, 1) + 0.41f) < 0.001);

        assert(fabs(g1(0, 0) + 0.00495f) < 0.001);
        assert(fabs(g1(0, 1) + 0.00997f) < 0.001);

        loadBuffer.Clear();
        layer.SaveWeights(loadBuffer);
        assert(loadBuffer.IsParamExist<CategoryTags::Matrix>("root/param"));
        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }
    
    void test_weight_layer3()
    {
        cout << "Test weight layer case 3 ...\t" << flush;
        using RootLayer = MakeTrainLayer<WeightLayer, CommonInputMap, PUpdate, PFeedbackOutput, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        static_assert(RootLayer::IsFeedbackOutput);
        static_assert(RootLayer::IsUpdate);

        RootLayer layer("root", 2, 2);

        Matrix<CheckElement, CheckDevice> w(2, 2);
        w.SetValue(0, 0, 1.1f); w.SetValue(0, 1, 3.1f);
        w.SetValue(1, 0, 0.1f); w.SetValue(1, 1, 1.17f);

        auto initializer = MakeInitializer<CheckElement>();
        initializer.SetParam("root/param", w);
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        layer.Init(initializer, loadBuffer);

        Matrix<CheckElement, CheckDevice> input(1, 2);
        input.SetValue(0, 0, 0.999f);
        input.SetValue(0, 1, 0.0067f);

        auto wi = LayerInputCont<RootLayer>().Set<LayerInput>(input);

        LayerNeutralInvariant(layer);
        auto out = layer.FeedForward(wi);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(fabs(res(0, 0) - 1.0996f) < 0.001);
        assert(fabs(res(0, 1) - 3.1047f) < 0.001);

        Matrix<CheckElement, CheckDevice> g(1, 2);
        g.SetValue(0, 0, 0.0469f);
        g.SetValue(0, 1, -0.0394f);
        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(g));
        auto fbOut = Evaluate(out_grad.Get<LayerInput>());
        assert(fabs(fbOut(0, 0) + 0.07055) < 0.001);
        assert(fabs(fbOut(0, 1) + 0.041408f) < 0.001);

        GradCollector<CheckElement, CheckDevice> grad_collector;
        layer.GradCollect(grad_collector);
        auto& gradCont = grad_collector.GetContainer<CategoryTags::Matrix>();
        assert(gradCont.size() == 1);

        auto w1 = gradCont.begin()->second.Weight();
        
        auto handle2 = gradCont.begin()->second.Grad().EvalRegister();
        EvalPlan<CheckDevice>::Inst().Eval();
        auto g1 = handle2.Data();
        
        assert(fabs(w1(0, 0) - 1.1) < 0.001);
        assert(fabs(w1(1, 0) - 0.1) < 0.001);
        assert(fabs(w1(0, 1) - 3.1) < 0.001);
        assert(fabs(w1(1, 1) - 1.17) < 0.001);

        assert(fabs(g1(0, 0) - 0.0468531) < 0.001);
        assert(fabs(g1(1, 0) - 0.00031423) < 0.001);
        assert(fabs(g1(0, 1) + 0.0393606) < 0.001);
        assert(fabs(g1(1, 1) + 0.00026398) < 0.001);

        loadBuffer.Clear();
        layer.SaveWeights(loadBuffer);
        assert(loadBuffer.IsParamExist<CategoryTags::Matrix>("root/param"));

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
    
    void test_weight_layer4()
    {
        cout << "Test weight layer case 4 ...\t" << flush;
        using RootLayer = MakeTrainLayer<WeightLayer, CommonInputMap, PUpdate, PFeedbackOutput, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        static_assert(RootLayer::IsFeedbackOutput);
        static_assert(RootLayer::IsUpdate);

        RootLayer layer("root", 8, 4);

        auto w = GenTensor<CheckElement>(0.1f, 0.5f, 8, 4);
    
        auto initializer = MakeInitializer<CheckElement>();
        initializer.SetParam("root/param", w);
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        layer.Init(initializer, loadBuffer);

        vector<Matrix<CheckElement, CheckDevice>> op_in;
        vector<Matrix<CheckElement, CheckDevice>> op_grad;

        for (int loop_count = 0; loop_count < 10; ++loop_count)
        {
            auto input = GenTensor<CheckElement>(loop_count * 0.1f, -0.3f, 1, 8);
            op_in.push_back(input);

            auto out = layer.FeedForward(LayerInputCont<RootLayer>().Set<LayerInput>(input));
            auto check = Dot(input, w);

            auto handle1 = out.Get<LayerOutput>().EvalRegister();
            auto handle2 = check.EvalRegister();
            EvalPlan<CheckDevice>::Inst().Eval();

            auto res = handle1.Data();
            assert(res.Shape() == Shape(1, 4));
            auto c = handle2.Data();

            for (size_t i = 0; i < 4; ++i)
            {
                assert(fabs(res(0, i) - c(0, i)) <= 0.0001f);
            }
        }

        for (int loop_count = 9; loop_count >= 0; --loop_count)
        {
            auto grad = GenTensor<CheckElement>(loop_count * 0.2f, -0.1f, 1, 4);
            op_grad.push_back(grad);
            auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));
            auto check = Dot(grad, Transpose(w));

            auto handle1 = out_grad.Get<LayerInput>().EvalRegister();
            auto handle2 = check.EvalRegister();
            EvalPlan<CheckDevice>::Inst().Eval();

            auto fbOut = handle1.Data();
            auto aimFbout = handle2.Data();
            assert(fbOut.Shape() == Shape(1, 8));

            for (size_t i = 0; i < 8; ++i)
            {
                assert(fabs(fbOut(0, i) - aimFbout(0, i)) < 0.0001);
            }
        }
        reverse(op_grad.begin(), op_grad.end());
        
        GradCollector<CheckElement, CheckDevice> grad_collector;
        layer.GradCollect(grad_collector);
        auto& gradCont = grad_collector.GetContainer<CategoryTags::Matrix>();
        assert(gradCont.size() == 1);
        
        auto w1 = gradCont.begin()->second.Weight();
        auto aim = Evaluate(Dot(Transpose(op_in[0]), op_grad[0]));
        for (int loop_count = 1; loop_count < 10; ++loop_count)
        {
            aim = Evaluate(aim + Dot(Transpose(op_in[loop_count]), op_grad[loop_count]));
        }

        auto g1 = Evaluate(gradCont.begin()->second.Grad());

        for (size_t i = 0; i < 8; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                assert(fabs(aim(i, j) - g1(i, j)) < 0.0001f);
            }
        }

        loadBuffer.Clear();
        layer.SaveWeights(loadBuffer);
        assert(loadBuffer.IsParamExist<CategoryTags::Matrix>("root/param"));

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
    
    struct RootFiller;
    void test_weight_layer5()
    {
        cout << "Test weight layer case 5 ...\t" << flush;
        using RootLayer = MakeTrainLayer<WeightLayer, CommonInputMap, PUpdate, PFeedbackOutput, PInitializerIs<RootFiller>, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        RootLayer layer("root", 800, 400);

        auto initializer = MakeInitializer<CheckElement>(InitializerKV<RootFiller>(UniformFiller{-1, 1}));
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        LayerInit(layer, initializer, loadBuffer);
        assert(loadBuffer.IsParamExist<CategoryTags::Matrix>("root/param"));

        auto& val = *(loadBuffer.TryGet<CategoryTags::Matrix>("root/param"));
    
        float mean = 0;
        for (size_t i = 0; i < val.Shape()[0]; ++i)
        {
            for (size_t j = 0; j < val.Shape()[1]; ++j)
            {
                mean += val(i, j);
            }
        }
        mean /= val.Shape().Count();
    
        float var = 0;
        for (size_t i = 0; i < val.Shape()[0]; ++i)
        {
            for (size_t j = 0; j < val.Shape()[1]; ++j)
            {
                var += (val(i, j) - mean) * (val(i, j) - mean);
            }
        }
        var /= val.Shape().Count();
    
        // should be about 0, 0.333
        cout << "mean-delta = " << fabs(mean) << " Variance-delta = " << fabs(var-0.333) << ' ';
        cout << "done" << endl;
    }
    
    void test_weight_layer6()
    {
        cout << "Test weight layer case 6 ...\t" << flush;
        using RootLayer = MakeTrainLayer<WeightLayer, CommonInputMap, PUpdate, PFeedbackOutput, PInitializerIs<RootFiller>, PParamTypeIs<Matrix<CheckElement, CheckDevice>>>;
        RootLayer layer("root", 400, 200);

        auto initializer = MakeInitializer<CheckElement>(InitializerKV<RootFiller>(UniformFiller{-1.5, 1.5}));
        LoadBuffer<CheckElement, CheckDevice> loadBuffer;
        LayerInit(layer, initializer, loadBuffer);
        assert(loadBuffer.IsParamExist<CategoryTags::Matrix>("root/param"));

        auto& val = *(loadBuffer.TryGet<CategoryTags::Matrix>("root/param"));
    
        float mean = 0;
        for (size_t i = 0; i < val.Shape()[0]; ++i)
        {
            for (size_t j = 0; j < val.Shape()[1]; ++j)
            {
                mean += val(i, j);
            }
        }
        mean /= val.Shape().Count();
    
        float var = 0;
        for (size_t i = 0; i < val.Shape()[0]; ++i)
        {
            for (size_t j = 0; j < val.Shape()[1]; ++j)
            {
                var += (val(i, j) - mean) * (val(i, j) - mean);
            }
        }
        var /= val.Shape().Count();
        // should be about 0, 0.75
        cout << "mean-delta = " << fabs(mean) << " Variance-delta = " << fabs(var-0.75) << ' ';

        cout << "done" << endl;
    }
}

namespace Test::Layer::Compose
{
    void test_weight_layer()
    {
        test_weight_layer1();
        test_weight_layer2();
        test_weight_layer3();
        test_weight_layer4();
        test_weight_layer5();
        test_weight_layer6();
    }
}