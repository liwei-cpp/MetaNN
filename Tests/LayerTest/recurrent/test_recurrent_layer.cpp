#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <data_gen.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
    struct AddSublayer;

    namespace NSAddWrapLayer
    {
        using Topology = ComposeTopology<Sublayer<AddSublayer, AddLayer>,
                                         InConnect<LayerInput, AddSublayer, LeftOperand>,
                                         InConnect<Previous<LayerOutput>, AddSublayer, RightOperand>,
                                         OutConnect<AddSublayer, LayerOutput, LayerOutput>>;
        template <typename TInputMap, typename TPolicies>
        using Base = ComposeKernel<LayerPortSet<LayerInput, Previous<LayerOutput>>,
                                   LayerPortSet<LayerOutput>, TInputMap, TPolicies, Topology>;

    }

    template <typename TInputs, typename TPolicies>
    class AddWrapLayer : public NSAddWrapLayer::Base<TInputs, TPolicies>
    {
        using TBase = NSAddWrapLayer::Base<TInputs, TPolicies>;
    public:
        AddWrapLayer(std::string p_name)
            : TBase(TBase::CreateSublayers().template Set<AddSublayer>(std::move(p_name)))
        { }
    };
}

namespace
{
    using TrivalInputMap = LayerIOMap<LayerKV<LayerInput, Matrix<CheckElement, CheckDevice>>,
                                      LayerKV<Previous<LayerOutput>, Matrix<CheckElement, CheckDevice>>>;
    void test_recurrent_layer1()
    {
        cout << "Test recurrent layer case 1 (trival)...\t";
        using RootLayer = MakeTrainLayer<RecurrentLayer, TrivalInputMap, PActFuncIs<AddWrapLayer>>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);

        RootLayer layer("root");
        auto input = GenMatrix<CheckElement>(5, 7, -1, 0.01f);
        auto prev = GenMatrix<CheckElement>(5, 7, 2, -0.03f);
        auto outputCont = layer.FeedForward(LayerInputCont<RootLayer>()
                                            .Set<LayerInput>(input)
                                            .Set<Previous<LayerOutput>>(prev));
        auto res = Evaluate(outputCont.Get<LayerOutput>());
        
        assert(res.Shape() == input.Shape());
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res(j, k) - input(j, k) - prev(j, k)) < 0.001f);
            }
        }
        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }
    
    void test_recurrent_layer2()
    {
        cout << "Test recurrent layer case 2 (trival, feedout)...\t";
        using RootLayer = MakeTrainLayer<RecurrentLayer, TrivalInputMap, PActFuncIs<AddWrapLayer>, PFeedbackOutput>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(RootLayer::IsFeedbackOutput);

        RootLayer layer("root");
        auto input = GenMatrix<CheckElement>(5, 7, -1, 0.01f);
        auto prev = GenMatrix<CheckElement>(5, 7, 2, -0.03f);
        auto outputCont = layer.FeedForward(LayerInputCont<RootLayer>()
                                            .Set<LayerInput>(input)
                                            .Set<Previous<LayerOutput>>(prev));
        auto res = Evaluate(outputCont.Get<LayerOutput>());
        
        assert(res.Shape() == input.Shape());
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res(j, k) - input(j, k) - prev(j, k)) < 0.001f);
            }
        }
        
        auto grad = GenMatrix<CheckElement>(5, 7, 0, 0.13f);
        auto gradOutputCont = layer.FeedBackward(LayerOutputCont<RootLayer>()
                                                 .Set<LayerOutput>(grad));
        auto grad_input = Evaluate(gradOutputCont.Get<LayerInput>());
        auto grad_prev = Evaluate(gradOutputCont.Get<Previous<LayerOutput>>());
        assert(grad_input.Shape() == grad.Shape());
        assert(grad_prev.Shape() == grad.Shape());
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(grad_input(j, k) - grad(j, k)) < 0.001f);
                assert(fabs(grad_prev(j, k) - grad(j, k)) < 0.001f);
            }
        }
        
        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }
    
    using SeqInputMap = LayerIOMap<LayerKV<LayerInput, MatrixSequence<CheckElement, CheckDevice>>,
                                   LayerKV<Previous<LayerOutput>, Matrix<CheckElement, CheckDevice>>>;
        
    void test_recurrent_layer3()
    {
        cout << "Test recurrent layer case 3 (non-trival)...\t";
        using RootLayer = MakeTrainLayer<RecurrentLayer, SeqInputMap, PActFuncIs<AddWrapLayer>>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);
        
        RootLayer layer("root");
        auto input = GenMatrixSequence<CheckElement>(3, 5, 7, -1, 0.01f);
        auto prev = GenMatrix<CheckElement>(5, 7, 2, -0.03f);
        auto outputCont = layer.FeedForward(LayerInputCont<RootLayer>()
                                            .Set<LayerInput>(input)
                                            .Set<Previous<LayerOutput>>(prev));
        auto res = Evaluate(outputCont.Get<LayerOutput>());
        
        assert(res.Shape() == input.Shape());
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res[0](j, k) - input[0](j, k) - prev(j, k)) < 0.001f);
            }
        }
        
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res[1](j, k) - input[1](j, k) - res[0](j, k)) < 0.001f);
            }
        }
        
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res[2](j, k) - input[2](j, k) - res[1](j, k)) < 0.001f);
            }
        }
        
        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }
    
    void test_recurrent_layer4()
    {
        cout << "Test recurrent layer case 4 (non-trival, feedout)...\t";
        using RootLayer = MakeTrainLayer<RecurrentLayer, SeqInputMap, PActFuncIs<AddWrapLayer>, PFeedbackOutput>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(RootLayer::IsFeedbackOutput);
        
        RootLayer layer("root");
        auto input = GenMatrixSequence<CheckElement>(3, 5, 7, -1, 0.01f);
        auto prev = GenMatrix<CheckElement>(5, 7, 2, -0.03f);
        auto outputCont = layer.FeedForward(LayerInputCont<RootLayer>()
                                            .Set<LayerInput>(input)
                                            .Set<Previous<LayerOutput>>(prev));
        auto res = Evaluate(outputCont.Get<LayerOutput>());
        
        assert(res.Shape() == input.Shape());
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res[0](j, k) - input[0](j, k) - prev(j, k)) < 0.001f);
            }
        }
        
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res[1](j, k) - input[1](j, k) - res[0](j, k)) < 0.001f);
            }
        }
        
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res[2](j, k) - input[2](j, k) - res[1](j, k)) < 0.001f);
            }
        }
        
        auto grad = GenMatrixSequence<CheckElement>(3, 5, 7, 0, 0.13f);
        auto gradOutputCont = layer.FeedBackward(LayerOutputCont<RootLayer>()
                                                 .Set<LayerOutput>(grad));
        auto grad_input = Evaluate(gradOutputCont.Get<LayerInput>());
        auto grad_prev = Evaluate(gradOutputCont.Get<Previous<LayerOutput>>());
        assert(grad_input.Shape() == input.Shape());
        assert(grad_prev.Shape() == prev.Shape());
        
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(grad_input[2](j, k) - grad[2](j, k)) < 0.001f);
                assert(fabs(grad_input[1](j, k) - grad[1](j, k) - grad[2](j, k)) < 0.001f);
                assert(fabs(grad_input[0](j, k) - grad[0](j, k) - grad[1](j, k) - grad[2](j, k)) < 0.001f);
                assert(fabs(grad_prev(j, k) - grad_input[0](j, k)) < 0.001f);
            }
        }
        
        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }
    
    void test_recurrent_layer5()
    {
        cout << "Test recurrent layer case 5 (no io map)...\t";
        using RootLayer = MakeInferLayer<RecurrentLayer, PActFuncIs<AddWrapLayer>>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);
        
        RootLayer layer("root");
        auto input = GenMatrixSequence<CheckElement>(3, 5, 7, -1, 0.01f);
        auto prev = GenMatrix<CheckElement>(5, 7, 2, -0.03f);
        auto outputCont = layer.FeedForward(LayerInputCont<RootLayer>()
                                            .Set<LayerInput>(input)
                                            .Set<Previous<LayerOutput>>(prev));
        auto res = Evaluate(outputCont.Get<LayerOutput>());
        
        assert(res.Shape() == input.Shape());
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res[0](j, k) - input[0](j, k) - prev(j, k)) < 0.001f);
            }
        }
        
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res[1](j, k) - input[1](j, k) - res[0](j, k)) < 0.001f);
            }
        }
        
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res[2](j, k) - input[2](j, k) - res[1](j, k)) < 0.001f);
            }
        }
        
        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }
}

namespace Test::Layer::Recurrent
{
    void test_recurrent_layer()
    {
        test_recurrent_layer1();
        test_recurrent_layer2();
        test_recurrent_layer3();
        test_recurrent_layer4();
        test_recurrent_layer5();
    }
}