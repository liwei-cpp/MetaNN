#include <MetaNN/meta_nn2.h>
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
    
    struct AddSublayer1;
    struct AddSublayer2;

    namespace NSAddWrapLayer2
    {
        using Topology = ComposeTopology<Sublayer<AddSublayer1, AddLayer>,
                                         Sublayer<AddSublayer2, AddLayer>,
                                         InConnect<LeftOperand, AddSublayer1, LeftOperand>,
                                         InConnect<RightOperand, AddSublayer2, LeftOperand>,
                                         InConnect<Previous<LayerOutput>, AddSublayer1, RightOperand>,
                                         InternalConnect<AddSublayer1, LayerOutput, AddSublayer2, RightOperand>,
                                         OutConnect<AddSublayer2, LayerOutput, LayerOutput>>;
        template <typename TInputMap, typename TPolicies>
        using Base = ComposeKernel<LayerPortSet<LeftOperand, RightOperand, Previous<LayerOutput>>,
                                   LayerPortSet<LayerOutput>, TInputMap, TPolicies, Topology>;

    }

    template <typename TInputs, typename TPolicies>
    class AddWrapLayer2 : public NSAddWrapLayer2::Base<TInputs, TPolicies>
    {
        using TBase = NSAddWrapLayer2::Base<TInputs, TPolicies>;
    public:
        AddWrapLayer2(std::string p_name)
            : TBase(TBase::CreateSublayers().template Set<AddSublayer1>(std::move(p_name + "/1"))
                                            .template Set<AddSublayer2>(std::move(p_name + "/2")))
        { }
    };
}

namespace
{
    void test_recurrent_layer1()
    {
        cout << "Test recurrent layer case 1...\t";
        using RootLayer = MakeInferLayer<RecurrentLayer, PSeqIDsAre<SeqID<LayerInput, 0>>, PActFuncIs<AddWrapLayer>>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);
        
        RootLayer layer("root");
        auto input = GenTensor<CheckElement>(-1, 0.01f, 3, 5, 7);
        auto prev = GenTensor<CheckElement>(2, -0.03f, 5, 7);
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
    
    using SeqInputMap = LayerIOMap<LayerKV<LayerInput, Tensor<CheckElement, CheckDevice, 3>>,
                                   LayerKV<Previous<LayerOutput>, Matrix<CheckElement, CheckDevice>>>;
        
    void test_recurrent_layer2()
    {
        cout << "Test recurrent layer case 2 (non-trival)...\t";
        using RootLayer = MakeTrainLayer<RecurrentLayer, SeqInputMap, PSeqIDsAre<SeqID<LayerInput, 0>>, PActFuncIs<AddWrapLayer>>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);
        
        RootLayer layer("root");
        auto input = GenTensor<CheckElement>(-1, 0.01f, 3, 5, 7);
        auto prev = GenTensor<CheckElement>(2, -0.03f, 5, 7);
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
    
    void test_recurrent_layer3()
    {
        cout << "Test recurrent layer case 3 (non-trival, feedout)...\t";
        using RootLayer = MakeTrainLayer<RecurrentLayer, SeqInputMap, PActFuncIs<AddWrapLayer>, PSeqIDsAre<SeqID<LayerInput, 0>>, PFeedbackOutput>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(RootLayer::IsFeedbackOutput);
        
        RootLayer layer("root");
        auto input = GenTensor<CheckElement>(-1, 0.01f, 3, 5, 7);
        auto prev = GenTensor<CheckElement>(2, -0.03f, 5, 7);
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
        
        auto grad = GenTensor<CheckElement>(0, 0.13f, 3, 5, 7);
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
                assert(fabs(grad_input(0, j, k) - grad(0, j, k) - grad(1, j, k) - grad(2, j, k)) < 0.001f);
                assert(fabs(grad_prev(j, k) - grad_input(0, j, k)) < 0.001f);
            }
        }
        
        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }
    
    void test_recurrent_layer4()
    {
        cout << "Test recurrent layer case 4 (2 inputs)...\t";
        using RootLayer = MakeInferLayer<RecurrentLayer, PSeqIDsAre<SeqID<LeftOperand, 0>>, PActFuncIs<AddWrapLayer2>>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);
        
        RootLayer layer("root");
        auto lInput = GenTensor<CheckElement>(-1, 0.01f, 3, 5, 7);
        auto rInput = GenTensor<CheckElement>(-1, 0.01f, 5, 7);
        auto prev = GenTensor<CheckElement>(2, -0.03f, 5, 7);
        auto outputCont = layer.FeedForward(LayerInputCont<RootLayer>()
                                            .Set<LeftOperand>(lInput)
                                            .Set<RightOperand>(rInput)
                                            .Set<Previous<LayerOutput>>(prev));
        auto res = Evaluate(outputCont.Get<LayerOutput>());
        
        assert(res.Shape() == Shape(3, 5, 7));
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res(0, j, k) - lInput(0, j, k) - rInput(j, k) - prev(j, k)) < 0.001f);
            }
        }
        
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res(1, j, k) - lInput(1, j, k) - rInput(j, k) - res(0, j, k)) < 0.001f);
            }
        }
        
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res(2, j, k) - lInput(2, j, k) - rInput(j, k) - res(1, j, k)) < 0.001f);
            }
        }
        
        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }
    
    void test_recurrent_layer5()
    {
        cout << "Test recurrent layer case 5 (2 inputs, non-zero seq ID)...\t";
        using RootLayer = MakeInferLayer<RecurrentLayer, PSeqIDsAre<SeqID<LeftOperand, 1>>, PActFuncIs<AddWrapLayer2>>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);
        
        RootLayer layer("root");
        auto lInput = GenTensor<CheckElement>(-1, 0.01f, 5, 3, 7);
        auto rInput = GenTensor<CheckElement>(-1, 0.01f, 5, 7);
        auto prev = GenTensor<CheckElement>(2, -0.03f, 5, 7);
        auto outputCont = layer.FeedForward(LayerInputCont<RootLayer>()
                                            .Set<LeftOperand>(lInput)
                                            .Set<RightOperand>(rInput)
                                            .Set<Previous<LayerOutput>>(prev));
        auto res = Evaluate(outputCont.Get<LayerOutput>());
        
        assert(res.Shape() == Shape(3, 5, 7));
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res(0, j, k) - lInput(j, 0, k) - rInput(j, k) - prev(j, k)) < 0.001f);
            }
        }
        
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res(1, j, k) - lInput(j, 1, k) - rInput(j, k) - res(0, j, k)) < 0.001f);
            }
        }
        
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res(2, j, k) - lInput(j, 2, k) - rInput(j, k) - res(1, j, k)) < 0.001f);
            }
        }
        
        LayerNeutralInvariant(layer);

        cout << "done" << endl;
    }
    
    void test_recurrent_layer6()
    {
        cout << "Test recurrent layer case 6 (2 inputs, non-zero seq ID)...\t";
        using RootLayer = MakeInferLayer<RecurrentLayer, PSeqIDsAre<SeqID<LeftOperand, 1>, SeqID<RightOperand, 2>>,
                                                         PActFuncIs<AddWrapLayer2>>;
        static_assert(!RootLayer::IsUpdate);
        static_assert(!RootLayer::IsFeedbackOutput);
        
        RootLayer layer("root");
        auto lInput = GenTensor<CheckElement>(-1, 0.01f, 5, 3, 7);
        auto rInput = GenTensor<CheckElement>(-1, 0.01f, 5, 7, 3);
        auto prev = GenTensor<CheckElement>(2, -0.03f, 5, 7);
        auto outputCont = layer.FeedForward(LayerInputCont<RootLayer>()
                                            .Set<LeftOperand>(lInput)
                                            .Set<RightOperand>(rInput)
                                            .Set<Previous<LayerOutput>>(prev));
        auto res = Evaluate(outputCont.Get<LayerOutput>());
        
        assert(res.Shape() == Shape(3, 5, 7));
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res(0, j, k) - lInput(j, 0, k) - rInput(j, k, 0) - prev(j, k)) < 0.001f);
            }
        }
        
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res(1, j, k) - lInput(j, 1, k) - rInput(j, k, 1) - res(0, j, k)) < 0.001f);
            }
        }
        
        for (size_t j = 0; j < 5; ++j)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                assert(fabs(res(2, j, k) - lInput(j, 2, k) - rInput(j, k, 2) - res(1, j, k)) < 0.001f);
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
        test_recurrent_layer6();
    }
}