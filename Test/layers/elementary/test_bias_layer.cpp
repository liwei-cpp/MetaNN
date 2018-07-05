#include <MetaNN/meta_nn.h>
#include "../../facilities/data_gen.h"
#include <cassert>
#include <iostream>
#include <map>
using namespace MetaNN;
using namespace std;

namespace
{
void test_bias_layer1()
{
    cout << "Test bias layer case 1 ...\t";
    using RootLayer = InjectPolicy<BiasLayer>;
    static_assert(!RootLayer::IsUpdate, "Test Error");
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 1);
    
    auto initializer = MakeInitializer<float>();
    auto weight = GenMatrix<float>(2, 1, 1, 0.1f);
    initializer.SetMatrix("root", weight);
    map<string, Matrix<float, DeviceTags::CPU>> params;
    layer.Init(initializer, params);

    auto input = GenMatrix<float>(2, 1, 0.5f, -0.1f);
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

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root") != params.end());

    LayerNeutralInvariant(layer);
    cout << "done" << endl;
}

void test_bias_layer2()
{
    cout << "Test bias layer case 2 ...\t";
    using RootLayer = InjectPolicy<BiasLayer>;
    static_assert(!RootLayer::IsUpdate, "Test Error");
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");

    RootLayer layer("root", 1, 2);
    
    auto initializer = MakeInitializer<float>();
    auto weight = GenMatrix<float>(1, 2, 1, 0.1f);
    initializer.SetMatrix("root", weight);
    map<string, Matrix<float, DeviceTags::CPU>> params;
    layer.Init(initializer, params);
    
    auto input = GenMatrix<float>(1, 2, 0.5f, -0.1f);
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

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root") != params.end());

    cout << "done" << endl;
}

void test_bias_layer3()
{
    cout << "Test bias layer case 3 ...\t";
    using RootLayer = InjectPolicy<BiasLayer, PUpdate>;
    static_assert(RootLayer::IsUpdate, "Test Error");
    static_assert(!RootLayer::IsFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 1);
    
    Matrix<float, DeviceTags::CPU> w(2, 1);
    w.SetValue(0, 0, -0.48f);
    w.SetValue(1, 0, -0.13f);

    auto initializer = MakeInitializer<float>();
    initializer.SetMatrix("root", w);
    map<string, Matrix<float, DeviceTags::CPU>> params;
    layer.Init(initializer, params);
    
    Matrix<float, DeviceTags::CPU> input(2, 1);
    input.SetValue(0, 0, -0.27f);
    input.SetValue(1, 0, -0.41f);

    auto bi = LayerIO::Create().Set<LayerIO>(input);

    LayerNeutralInvariant(layer);
    auto out = layer.FeedForward(bi);
    auto res = Evaluate(out.Get<LayerIO>());
    assert(fabs(res(0, 0) + 0.27f + 0.48f) < 0.001);
    assert(fabs(res(1, 0) + 0.41f + 0.13f) < 0.001);

    Matrix<float, DeviceTags::CPU> g(2, 1);
    g.SetValue(0, 0, -0.0495f);
    g.SetValue(1, 0, -0.0997f);

    auto fbIn = LayerIO::Create().Set<LayerIO>(g);
    auto out_grad = layer.FeedBackward(fbIn);

    GradCollector<float, DeviceTags::CPU> grad_collector;
    layer.GradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto gcit = grad_collector.begin();
    auto claps = Collapse(gcit->grad);

    auto handle1 = gcit->weight.EvalRegister();
    auto handle2 = claps.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();

    auto w1 = handle1.Data();
    auto g1 = handle2.Data();

    assert(fabs(w1(0, 0) + 0.48f) < 0.001);
    assert(fabs(w1(1, 0) + 0.13f) < 0.001);
    assert(fabs(g1(0, 0) + 0.0495f) < 0.001);
    assert(fabs(g1(1, 0) + 0.0997f) < 0.001);
    LayerNeutralInvariant(layer);

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root") != params.end());

    cout << "done" << endl;
}

void test_bias_layer4()
{
    cout << "Test bias layer case 4 ...\t";
    using RootLayer = InjectPolicy<BiasLayer, PUpdate, PFeedbackOutput>;
    static_assert(RootLayer::IsUpdate, "Test Error");
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 1);
    
    Matrix<float, DeviceTags::CPU> w(2, 1);
    w.SetValue(0, 0, -0.48f);
    w.SetValue(1, 0, -0.13f);
    
    auto initializer = MakeInitializer<float>();
    initializer.SetMatrix("root", w);
    map<string, Matrix<float, DeviceTags::CPU>> params;
    layer.Init(initializer, params);
    
    Matrix<float, DeviceTags::CPU> input(2, 1);
    input.SetValue(0, 0, -0.27f);
    input.SetValue(1, 0, -0.41f);

    auto bi = LayerIO::Create().Set<LayerIO>(input);

    LayerNeutralInvariant(layer);
    auto out = layer.FeedForward(bi);
    auto res = Evaluate(out.Get<LayerIO>());
    assert(fabs(res(0, 0) + 0.27f + 0.48f) < 0.001);
    assert(fabs(res(1, 0) + 0.41f + 0.13f) < 0.001);

    Matrix<float, DeviceTags::CPU> g(2, 1);
    g.SetValue(0, 0, -0.0495f);
    g.SetValue(1, 0, -0.0997f);

    auto fbIn = LayerIO::Create().Set<LayerIO>(g);
    auto out_grad = layer.FeedBackward(fbIn);
    auto fbOut = Evaluate(out_grad.Get<LayerIO>());

    assert(fabs(fbOut(0, 0) + 0.0495f) < 0.001);
    assert(fabs(fbOut(1, 0) + 0.0997f) < 0.001);

    GradCollector<float, DeviceTags::CPU> grad_collector;
    layer.GradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto gcit = grad_collector.begin();
    auto claps = Collapse(gcit->grad);

    auto handle1 = gcit->weight.EvalRegister();
    auto handle2 = claps.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();

    auto w1 = handle1.Data();
    auto g1 = handle2.Data();

    assert(fabs(w1(0, 0) + 0.48f) < 0.001);
    assert(fabs(w1(1, 0) + 0.13f) < 0.001);
    assert(fabs(g1(0, 0) + 0.0495f) < 0.001);
    assert(fabs(g1(1, 0) + 0.0997f) < 0.001);
    LayerNeutralInvariant(layer);

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root") != params.end());

    cout << "done" << endl;
}

void test_bias_layer5()
{
    cout << "Test bias layer case 5 ...\t";
    using RootLayer = InjectPolicy<BiasLayer, PUpdate, PFeedbackOutput>;
    static_assert(RootLayer::IsUpdate, "Test Error");
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");

    RootLayer layer("root", 2, 1);
    
    Matrix<float, DeviceTags::CPU> w(2, 1);
    w.SetValue(0, 0, -0.48f);
    w.SetValue(1, 0, -0.13f);
    
    auto initializer = MakeInitializer<float>();
    initializer.SetMatrix("root", w);
    map<string, Matrix<float, DeviceTags::CPU>> params;
    layer.Init(initializer, params);

    Matrix<float, DeviceTags::CPU> input(2, 1);
    input.SetValue(0, 0, -0.27f);
    input.SetValue(1, 0, -0.41f);

    auto bi = LayerIO::Create().Set<LayerIO>(input);

    LayerNeutralInvariant(layer);
    auto out = layer.FeedForward(bi);
    auto res = Evaluate(out.Get<LayerIO>());
    assert(fabs(res(0, 0) + 0.27f + 0.48f) < 0.001);
    assert(fabs(res(1, 0) + 0.41f + 0.13f) < 0.001);

    input = Matrix<float, DeviceTags::CPU>(2, 1);
    input.SetValue(0, 0, 1.27f);
    input.SetValue(1, 0, 2.41f);

    bi = LayerIO::Create().Set<LayerIO>(input);

    out = layer.FeedForward(bi);
    res = Evaluate(out.Get<LayerIO>());
    assert(fabs(res(0, 0) - 1.27f + 0.48f) < 0.001);
    assert(fabs(res(1, 0) - 2.41f + 0.13f) < 0.001);


    Matrix<float, DeviceTags::CPU> g(2, 1);
    g.SetValue(0, 0, -0.0495f);
    g.SetValue(1, 0, -0.0997f);

    auto fbIn = LayerIO::Create().Set<LayerIO>(g);
    auto out_grad = layer.FeedBackward(fbIn);
    auto fbOut = Evaluate(out_grad.Get<LayerIO>());

    assert(fabs(fbOut(0, 0) + 0.0495f) < 0.001);
    assert(fabs(fbOut(1, 0) + 0.0997f) < 0.001);

    g = Matrix<float, DeviceTags::CPU>(2, 1);
    g.SetValue(0, 0, 1.0495f);
    g.SetValue(1, 0, 2.3997f);

    fbIn = LayerIO::Create().Set<LayerIO>(g);
    out_grad = layer.FeedBackward(fbIn);
    fbOut = Evaluate(out_grad.Get<LayerIO>());

    assert(fabs(fbOut(0, 0) - 1.0495f) < 0.001);
    assert(fabs(fbOut(1, 0) - 2.3997f) < 0.001);


    GradCollector<float, DeviceTags::CPU> grad_collector;
    layer.GradCollect(grad_collector);
    assert(grad_collector.size() == 1);

    auto gcit = grad_collector.begin();
    auto claps = Collapse(gcit->grad);

    auto handle1 = gcit->weight.EvalRegister();
    auto handle2 = claps.EvalRegister();
    EvalPlan<DeviceTags::CPU>::Eval();

    auto w1 = handle1.Data();
    auto g1 = handle2.Data();

    assert(fabs(w1(0, 0) + 0.48f) < 0.001);
    assert(fabs(w1(1, 0) + 0.13f) < 0.001);
    assert(fabs(g1(0, 0) + 0.0495f - 1.0495f) < 0.001);
    assert(fabs(g1(1, 0) + 0.0997f - 2.3997f) < 0.001);
    LayerNeutralInvariant(layer);

    params.clear();
    layer.SaveWeights(params);
    assert(params.find("root") != params.end());

    cout << "done" << endl;
}

void test_bias_layer6()
{
    cout << "Test bias layer case 6 ...\t";
    using RootLayer = InjectPolicy<BiasLayer, PUpdate, PFeedbackOutput>;
    static_assert(RootLayer::IsUpdate, "Test Error");
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    
    RootLayer layer("root", 400);
    
    auto initializer = MakeInitializer<float, PInitializerIs<struct ConstantTag>>()
                            .SetFiller<ConstantTag>(ConstantFiller{0});
    map<string, Matrix<float, DeviceTags::CPU>> loader;
    layer.Init(initializer, loader);
    
    assert(loader.size() == 1);
    
    auto& val = loader.begin()->second;
    
    for (size_t i = 0; i < val.RowNum(); ++i)
    {
        for (size_t j = 0; j < val.ColNum(); ++j)
        {
            assert(fabs(val(i, j)) < 0.0001);
        }
    }
    cout << "done" << endl;
}

void test_bias_layer7()
{
    cout << "Test bias layer case 7 ...\t";
    using RootLayer = InjectPolicy<BiasLayer, PUpdate, PFeedbackOutput>;
    static_assert(RootLayer::IsUpdate, "Test Error");
    static_assert(RootLayer::IsFeedbackOutput, "Test Error");
    
    RootLayer layer("root", 400);
    
    auto initializer = MakeInitializer<float, PInitializerIs<struct ConstantTag>>()
                            .SetFiller<ConstantTag>(ConstantFiller{1.5});
    map<string, Matrix<float, DeviceTags::CPU>> loader;
    layer.Init(initializer, loader);
    
    assert(loader.size() == 1);
    
    auto& val = loader.begin()->second;
    
    for (size_t i = 0; i < val.RowNum(); ++i)
    {
        for (size_t j = 0; j < val.ColNum(); ++j)
        {
            assert(fabs(val(i, j) - 1.5) < 0.0001);
        }
    }
    cout << "done" << endl;
}
}

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
