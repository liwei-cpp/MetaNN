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
    using BatchInputMap = LayerIOMap<LayerKV<LayerInput, BatchMatrix<CheckElement, CheckDevice>>>;

    void test_batch_iter_layer1()
    {
        cout << "Test batch iterator layer case 1 (Tanh kernel, inference)...\t";
        using RootLayer = MakeInferLayer<BatchIterLayer, PActFuncIs<TanhLayer>>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);

        RootLayer layer("root");
        auto input = GenBatchMatrix<CheckElement>(3, 5, 7, -1, 0.01f);
        auto outputCont = layer.FeedForward(LayerInputCont<RootLayer>().Set<LayerInput>(input));
        auto res = Evaluate(outputCont.Get<LayerOutput>());
        
        assert(res.Shape() == input.Shape());
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    assert(fabs(res[i](j, k) - tanh(input[i](j, k))) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_batch_iter_layer2()
    {
        cout << "Test batch iterator layer case 2 (Tanh kernel, train, trival)...\t";
        using RootLayer = MakeTrainLayer<BatchIterLayer, CommonInputMap, PFeedbackOutput, PActFuncIs<TanhLayer>>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(RootLayer::IsFeedbackOutput);

        RootLayer layer("root");
        auto input = GenMatrix<CheckElement>(5, 7, -1, 0.01f);
        auto outputCont = layer.FeedForward(LayerInputCont<RootLayer>().Set<LayerInput>(input));
        auto res = Evaluate(outputCont.Get<LayerOutput>());
        
        assert(res.Shape() == input.Shape());
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res(j, k) - tanh(input(j, k))) < 0.001f);
            }
        }
        
        auto grad = GenMatrix<CheckElement>(5, 7, 0, 0.03f);
        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));
        auto fb = Evaluate(out_grad.Get<LayerInput>());
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(fb(j, k) - grad(j, k) * (1 - res(j, k) * res(j, k))) < 0.001);
            }
        }
        cout << "done" << endl;
    }
    
    void test_batch_iter_layer3()
    {
        cout << "Test batch iterator layer case 2 (Tanh kernel, train, non-trival)...\t";
        using RootLayer = MakeTrainLayer<BatchIterLayer, BatchInputMap, PFeedbackOutput, PActFuncIs<TanhLayer>>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(RootLayer::IsFeedbackOutput);

        RootLayer layer("root");
        auto input = GenBatchMatrix<CheckElement>(3, 5, 7, -1, 0.01f);
        auto outputCont = layer.FeedForward(LayerInputCont<RootLayer>().Set<LayerInput>(input));
        auto res = Evaluate(outputCont.Get<LayerOutput>());
        
        assert(res.Shape() == input.Shape());
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    assert(fabs(res[i](j, k) - tanh(input[i](j, k))) < 0.001f);
                }
            }
        }

        auto grad = GenBatchMatrix<CheckElement>(3, 5, 7, 0, 0.03f);
        auto out_grad = layer.FeedBackward(LayerOutputCont<RootLayer>().Set<LayerOutput>(grad));
        auto fb = Evaluate(out_grad.Get<LayerInput>());
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    assert(fabs(fb[i](j, k) - grad[i](j, k) * (1 - res[i](j, k) * res[i](j, k))) < 0.001);
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Layer::Compose
{
    void test_batch_iter_layer()
    {
        test_batch_iter_layer1();
        test_batch_iter_layer2();
        test_batch_iter_layer3();
    }
}