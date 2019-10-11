#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <data_gen.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
    struct Sublayer1; struct Sublayer2; struct Sublayer3; struct Sublayer4; struct Sublayer5; struct Sublayer6;
    struct Input1; struct Input2; struct Output1;
    void test_compose_kernel1()
    {
        cout << "Test compose kernel case 1...\t";

        using namespace MetaNN::NSComposeKernel;
        using namespace MetaNN::ContMetaFun;

        using check1 = SeparateClauses_<Sublayer<Sublayer1, AddLayer>,
                                        Sublayer<Sublayer2, MultiplyLayer>,
                                        Sublayer<Sublayer3, SigmoidLayer>,
                                        Sublayer<Sublayer4, TanhLayer>,
                                        Sublayer<Sublayer5, AddLayer>,
                                        InConnect<Input1, Sublayer1, LeftOperand>,
                                        InConnect<Input2, Sublayer1, RightOperand>,
                                        InConnect<Input1, Sublayer2, RightOperand>,
                                        InternalConnect<Sublayer1, LayerOutput, Sublayer2, LeftOperand>,
                                        InternalConnect<Sublayer2, LayerOutput, Sublayer3, LayerInput>,
                                        InternalConnect<Sublayer2, LayerOutput, Sublayer4, LayerInput>,
                                        InternalConnect<Sublayer3, LayerOutput, Sublayer5, LeftOperand>,
                                        InternalConnect<Sublayer4, LayerOutput, Sublayer5, RightOperand>,
                                        OutConnect<Sublayer5, LayerOutput, Output1>>;
        static_assert(std::is_same_v<check1::SublayerRes,
                                     ClauseSeq<Sublayer<Sublayer1, AddLayer>,
                                               Sublayer<Sublayer2, MultiplyLayer>,
                                               Sublayer<Sublayer3, SigmoidLayer>,
                                               Sublayer<Sublayer4, TanhLayer>,
                                               Sublayer<Sublayer5, AddLayer>>>);

        static_assert(std::is_same_v<check1::InConnectRes,
                                     ClauseSeq<InConnect<Input1, Sublayer1, LeftOperand>,
                                               InConnect<Input2, Sublayer1, RightOperand>,
                                               InConnect<Input1, Sublayer2, RightOperand>>>);

        static_assert(std::is_same_v<check1::OutConnectRes,
                                     ClauseSeq<OutConnect<Sublayer5, LayerOutput, Output1>>>);

        static_assert(std::is_same_v<ClauseRefine::InternalFMap<check1::InterConnectRes>,
                                     ClauseSeq<Helper::KVBinder<Sublayer1, Helper::ValueSequence<InternalConnect<Sublayer1, LayerOutput, Sublayer2, LeftOperand>>>,
                                               Helper::KVBinder<Sublayer2, Helper::ValueSequence<InternalConnect<Sublayer2, LayerOutput, Sublayer3, LayerInput>,
                                                                                                 InternalConnect<Sublayer2, LayerOutput, Sublayer4, LayerInput>>>,
                                               Helper::KVBinder<Sublayer3, Helper::ValueSequence<InternalConnect<Sublayer3, LayerOutput, Sublayer5, LeftOperand>>>,
                                               Helper::KVBinder<Sublayer4, Helper::ValueSequence<InternalConnect<Sublayer4, LayerOutput, Sublayer5, RightOperand>>>>>);

        static_assert(std::is_same_v<ClauseRefine::InternalBMap<check1::InterConnectRes>,
                                     ClauseSeq<Helper::KVBinder<Sublayer2, Helper::ValueSequence<InternalConnect<Sublayer1, LayerOutput, Sublayer2, LeftOperand>>>,
                                               Helper::KVBinder<Sublayer3, Helper::ValueSequence<InternalConnect<Sublayer2, LayerOutput, Sublayer3, LayerInput>>>,
                                               Helper::KVBinder<Sublayer4, Helper::ValueSequence<InternalConnect<Sublayer2, LayerOutput, Sublayer4, LayerInput>>>,
                                               Helper::KVBinder<Sublayer5, Helper::ValueSequence<InternalConnect<Sublayer3, LayerOutput, Sublayer5, LeftOperand>,
                                                                                                 InternalConnect<Sublayer4, LayerOutput, Sublayer5, RightOperand>>>>>);
        static_assert(std::is_same_v<ClauseRefine::InputFMap<check1::InConnectRes>,
                                     ClauseSeq<Helper::KVBinder<Sublayer1, Helper::ValueSequence<InConnect<Input1, Sublayer1, LeftOperand>,
                                                                                                 InConnect<Input2, Sublayer1, RightOperand>>>,
                                               Helper::KVBinder<Sublayer2, Helper::ValueSequence<InConnect<Input1, Sublayer2, RightOperand>>>>>);

        static_assert(std::is_same_v<ClauseRefine::OutputBMap<check1::OutConnectRes>,
                                     ClauseSeq<Helper::KVBinder<Sublayer5, Helper::ValueSequence<OutConnect<Sublayer5, LayerOutput, Output1>>>>>);

        static_assert(std::is_same_v<ClauseRefine::SublayerNameSet<check1::SublayerRes>,
                                     ClauseSeq<Sublayer1, Sublayer2, Sublayer3, Sublayer4, Sublayer5>>);

        static_assert(std::is_same_v<ClauseRefine::SublayerMap<check1::SublayerRes>,
                                     ClauseSeq<Helper::KVBinder<Sublayer1, Sublayer<Sublayer1, AddLayer>>,
                                               Helper::KVBinder<Sublayer2, Sublayer<Sublayer2, MultiplyLayer>>,
                                               Helper::KVBinder<Sublayer3, Sublayer<Sublayer3, SigmoidLayer>>,
                                               Helper::KVBinder<Sublayer4, Sublayer<Sublayer4, TanhLayer>>,
                                               Helper::KVBinder<Sublayer5, Sublayer<Sublayer5, AddLayer>>>>);

        static_assert(std::is_same_v<ClauseRefine::InternalInLayerSet<check1::InterConnectRes>,
                                     ClauseSeq<Sublayer2, Sublayer3, Sublayer4, Sublayer5>>);
        static_assert(std::is_same_v<ClauseRefine::InternalOutLayerSet<check1::InterConnectRes>,
                                     ClauseSeq<Sublayer1, Sublayer2, Sublayer3, Sublayer4>>);
        static_assert(std::is_same_v<ClauseRefine::InternalLayerSet<check1::InterConnectRes>,
                                     ClauseSeq<Sublayer2, Sublayer3, Sublayer4, Sublayer5, Sublayer1>>);

        static_assert(std::is_same_v<ClauseRefine::InternalInNamePortSet<check1::InterConnectRes>,
                                     ClauseSeq<Helper::Pair<Sublayer2, LeftOperand>,
                                               Helper::Pair<Sublayer3, LayerInput>,
                                               Helper::Pair<Sublayer4, LayerInput>,
                                               Helper::Pair<Sublayer5, LeftOperand>,
                                               Helper::Pair<Sublayer5, RightOperand>>>);

        static_assert(std::is_same_v<ClauseRefine::InputNamePortSet<check1::InterConnectRes, check1::InConnectRes>,
                                     ClauseSeq<Helper::Pair<Sublayer2, LeftOperand>,
                                               Helper::Pair<Sublayer3, LayerInput>,
                                               Helper::Pair<Sublayer4, LayerInput>,
                                               Helper::Pair<Sublayer5, LeftOperand>,
                                               Helper::Pair<Sublayer5, RightOperand>,
                                               Helper::Pair<Sublayer1, LeftOperand>,
                                               Helper::Pair<Sublayer1, RightOperand>,
                                               Helper::Pair<Sublayer2, RightOperand>>>);

        static_assert(std::is_same_v<ClauseRefine::InputLayerSet<check1::InConnectRes>,
                                     ClauseSeq<Sublayer1, Sublayer2>>);
        static_assert(std::is_same_v<ClauseRefine::OutputPortSet<check1::OutConnectRes>,
                                     ClauseSeq<Output1>>);
        static_assert(std::is_same_v<ClauseRefine::OutputLayerSet<check1::OutConnectRes>,
                                     ClauseSeq<Sublayer5>>);

        using check2 = SeparateClauses_<Sublayer<Sublayer1, AddLayer>,
                                        InConnect<Input1, Sublayer1, LeftOperand>,
                                        InConnect<Input2, Sublayer1, RightOperand>,
                                        Sublayer<Sublayer2, MultiplyLayer>,
                                        InternalConnect<Sublayer1, LayerOutput, Sublayer2, LeftOperand>,
                                        InternalConnect<Sublayer2, LayerOutput, Sublayer3, LayerInput>,
                                        Sublayer<Sublayer3, SigmoidLayer>,
                                        InConnect<Input1, Sublayer2, RightOperand>,
                                        InternalConnect<Sublayer2, LayerOutput, Sublayer4, LayerInput>,
                                        Sublayer<Sublayer4, TanhLayer>,
                                        InternalConnect<Sublayer3, LayerOutput, Sublayer5, LeftOperand>,
                                        OutConnect<Sublayer5, LayerOutput, Output1>,
                                        Sublayer<Sublayer5, AddLayer>,
                                        InternalConnect<Sublayer4, LayerOutput, Sublayer5, RightOperand>>;

        static_assert(std::is_same_v<check2::SublayerRes, check1::SublayerRes>);
        static_assert(std::is_same_v<ClauseRefine::InternalFMap<check1::InterConnectRes>,
                                     ClauseRefine::InternalFMap<check2::InterConnectRes>>);
        static_assert(std::is_same_v<ClauseRefine::InternalBMap<check1::InterConnectRes>,
                                     ClauseRefine::InternalBMap<check2::InterConnectRes>>);
        static_assert(std::is_same_v<check2::InConnectRes, check1::InConnectRes>);
        static_assert(std::is_same_v<check2::OutConnectRes, check1::OutConnectRes>);
        cout << "done" << endl;
    }
    
    void test_compose_kernel2()
    {
        cout << "Test compose kernel case 2...\t";

        using namespace MetaNN::NSComposeKernel;
        using namespace MetaNN::ContMetaFun;
        
        constexpr bool check1 = InternalTagInSublayer<ClauseSeq<InternalConnect<Sublayer1, LayerOutput, Sublayer2, LeftOperand>,
                                                                InternalConnect<Sublayer2, LayerOutput, Sublayer3, LayerInput>,
                                                                InternalConnect<Sublayer2, LayerOutput, Sublayer4, LayerInput>,
                                                                InternalConnect<Sublayer3, LayerOutput, Sublayer5, LeftOperand>,
                                                                InternalConnect<Sublayer4, LayerOutput, Sublayer5, RightOperand>>,
                                                      ClauseSeq<Sublayer1, Sublayer2, Sublayer3, Sublayer4, Sublayer5>>;
        static_assert(check1);

        constexpr bool check2 = InternalTagInSublayer<ClauseSeq<InternalConnect<Sublayer1, LayerOutput, Sublayer2, LeftOperand>,
                                                                InternalConnect<Sublayer2, LayerOutput, Sublayer3, LayerInput>,
                                                                InternalConnect<Sublayer2, LayerOutput, Sublayer4, LayerInput>,
                                                                InternalConnect<Sublayer3, LayerOutput, Sublayer5, LeftOperand>,
                                                                InternalConnect<Sublayer4, LayerOutput, Sublayer5, RightOperand>>,
                                                      ClauseSeq<Sublayer1, Sublayer2, Sublayer3, Sublayer4>>;
        static_assert(!check2);

        constexpr bool check3 = InternalTagInSublayer<ClauseSeq<InternalConnect<Sublayer1, LayerOutput, Sublayer2, LeftOperand>,
                                                                InternalConnect<Sublayer2, LayerOutput, Sublayer4, LayerInput>,
                                                                InternalConnect<Sublayer4, LayerOutput, Sublayer5, RightOperand>>,
                                                      ClauseSeq<Sublayer1, Sublayer2, Sublayer3, Sublayer4, Sublayer5>>;
        static_assert(check3);
        cout << "done" << endl;
    }
    
    void test_compose_kernel3()
    {
        cout << "Test compose kernel case 3...\t";
        using namespace MetaNN::NSComposeKernel;

        constexpr bool check1 = InputTagInSubLayer<ClauseSeq<InConnect<Input1, Sublayer1, LeftOperand>,
                                                             InConnect<Input2, Sublayer1, RightOperand>,
                                                             InConnect<Input1, Sublayer2, RightOperand>>,
                                                   ClauseSeq<Sublayer1, Sublayer2, Sublayer3, Sublayer4, Sublayer5>>;
        static_assert(check1);

        constexpr bool check2 = InputTagInSubLayer<ClauseSeq<InConnect<Input1, Sublayer1, LeftOperand>,
                                                             InConnect<Input2, Sublayer1, RightOperand>,
                                                             InConnect<Input1, Sublayer2, RightOperand>>,
                                                   ClauseSeq<Sublayer1, Sublayer3, Sublayer4, Sublayer5>>;
        static_assert(!check2);
        cout << "done" << endl;
    }
    
    void test_compose_kernel4()
    {
        cout << "Test compose kernel case 4...\t";
        using namespace MetaNN::NSComposeKernel;

        constexpr bool check1 = OutputTagInSubLayer<ClauseSeq<OutConnect<Sublayer5, LayerOutput, Output1>>,
                                                    ClauseSeq<Sublayer1, Sublayer3, Sublayer4, Sublayer5>>;
        static_assert(check1);
    
        constexpr bool check2 = OutputTagInSubLayer<ClauseSeq<OutConnect<Sublayer5, LayerOutput, Output1>>,
                                                    ClauseSeq<Sublayer1, Sublayer3, Sublayer4>>;
        static_assert(!check2);
        cout << "done" << endl;
    }
    
    void test_compose_kernel5()
    {
        cout << "Test compose kernel case 5...\t";
        using namespace MetaNN::NSComposeKernel;
        constexpr bool check1 = SublayerTagInOtherArrays<ClauseSeq<Sublayer1, Sublayer2, Sublayer3, Sublayer4, Sublayer5>,
                                                         ClauseSeq<Sublayer1, Sublayer2, Sublayer3, Sublayer4>,
                                                         ClauseSeq<Sublayer1, Sublayer2>,
                                                         ClauseSeq<Sublayer5>>;
        static_assert(check1);
    
        constexpr bool check2 = SublayerTagInOtherArrays<ClauseSeq<Sublayer1, Sublayer2, Sublayer3, Sublayer4, Sublayer5, Sublayer6>,
                                                         ClauseSeq<Sublayer1, Sublayer2, Sublayer3, Sublayer4>,
                                                         ClauseSeq<Sublayer1, Sublayer2>,
                                                         ClauseSeq<Sublayer5>>;
        static_assert(!check2);
        cout << "done" << endl;
    }
    
    void test_compose_kernel6()
    {
        cout << "Test compose kernel case 6...\t";
        using namespace MetaNN::NSComposeKernel;

        using InterConnects = ClauseSeq<InternalConnect<Sublayer2, LayerOutput, Sublayer3, LayerInput>,
                                        InternalConnect<Sublayer1, LayerOutput, Sublayer2, LeftOperand>,
                                        InternalConnect<Sublayer2, LayerOutput, Sublayer4, LayerInput>,
                                        InternalConnect<Sublayer3, LayerOutput, Sublayer5, LeftOperand>,
                                        InternalConnect<Sublayer4, LayerOutput, Sublayer5, RightOperand>>;

        using check1 = TopologicalOrdering_<ClauseSeq<Sublayer<Sublayer2, MultiplyLayer>,
                                                      Sublayer<Sublayer1, AddLayer>,
                                                      Sublayer<Sublayer5, AddLayer>,
                                                      Sublayer<Sublayer3, SigmoidLayer>,
                                                      Sublayer<Sublayer4, TanhLayer>>,
                                            InterConnects>::type;
        static_assert(std::is_same_v<check1,
                                     ClauseSeq<Sublayer1, Sublayer2, Sublayer3, Sublayer4, Sublayer5>>);
        cout << "done" << endl;
    }
    
    void test_compose_kernel7()
    {
        cout << "Test compose kernel case 7...\t";
        using namespace MetaNN::NSComposeKernel;
        using namespace MetaNN::ContMetaFun::Helper;
        
        using Input1Type = Matrix<CheckElement, CheckDevice>;
        using Input2Type = TrivalMatrix<Scalar<CheckElement, CheckDevice>>;
        using CheckInputs = LayerIOMap<LayerKV<Input1, Input1Type>,
                                       LayerKV<Input2, Input2Type>>;
        using CheckPolicyContainer = PolicyContainer<PFeedbackOutput,
                                                     SubPolicyContainer<Sublayer1, PNoUpdate, PFeedbackNoOutput>,
                                                     SubPolicyContainer<Sublayer2, PUpdate, PFeedbackNoOutput>,
                                                     SubPolicyContainer<Sublayer5, PFeedbackNoOutput>>;
        using CheckSublayerCont = ClauseSeq<Sublayer1, Sublayer2, Sublayer3, Sublayer4, Sublayer5>;
        using CheckSublayerClause = ClauseSeq<Sublayer<Sublayer1, AddLayer>,
                                              Sublayer<Sublayer2, MultiplyLayer>,
                                              Sublayer<Sublayer3, SigmoidLayer>,
                                              Sublayer<Sublayer4, TanhLayer>,
                                              Sublayer<Sublayer5, AddLayer>>;
        using CheckInConnects = ClauseSeq<InConnect<Input1, Sublayer1, LeftOperand>,
                                          InConnect<Input2, Sublayer1, RightOperand>,
                                          InConnect<Input1, Sublayer2, RightOperand>>;
        using CheckInterConnects = ClauseSeq<InternalConnect<Sublayer1, LayerOutput, Sublayer2, LeftOperand>,
                                             InternalConnect<Sublayer2, LayerOutput, Sublayer3, LayerInput>,
                                             InternalConnect<Sublayer2, LayerOutput, Sublayer4, LayerInput>,
                                             InternalConnect<Sublayer3, LayerOutput, Sublayer5, LeftOperand>,
                                             InternalConnect<Sublayer4, LayerOutput, Sublayer5, RightOperand>>;
        using CheckOutConnects = ClauseSeq<OutConnect<Sublayer5, LayerOutput, Output1>>;

        using Check = SublayerInstantiation_<CheckInputs, CheckPolicyContainer,
                                             CheckSublayerCont, CheckSublayerClause, CheckInConnects, CheckInterConnects, CheckOutConnects>;
        static_assert(std::is_same_v<Check::SublayerPolicy1,
                                     std::tuple<KVBinder<Sublayer1, PolicyContainer<PNoUpdate, PFeedbackNoOutput>>,
                                                KVBinder<Sublayer2, PolicyContainer<PUpdate, PFeedbackNoOutput>>,
                                                KVBinder<Sublayer3, PolicyContainer<PFeedbackOutput>>,
                                                KVBinder<Sublayer4, PolicyContainer<PFeedbackOutput>>,
                                                KVBinder<Sublayer5, PolicyContainer<PFeedbackNoOutput>>>>);

        static_assert(std::is_same_v<Check::SublayerPolicy2,
                                     std::tuple<KVBinder<Sublayer1, PolicyContainer<PNoUpdate, PFeedbackOutput>>,
                                                KVBinder<Sublayer2, PolicyContainer<PUpdate, PFeedbackOutput>>,
                                                KVBinder<Sublayer3, PolicyContainer<PFeedbackOutput>>,
                                                KVBinder<Sublayer4, PolicyContainer<PFeedbackOutput>>,
                                                KVBinder<Sublayer5, PolicyContainer<PFeedbackNoOutput>>>>);
        static_assert(std::is_same_v<Check::SublayerPolicyFinal,
                                     std::tuple<KVBinder<Sublayer1, PolicyContainer<PNoUpdate, PFeedbackOutput>>,
                                                KVBinder<Sublayer2, PolicyContainer<PUpdate, PFeedbackOutput>>,
                                                KVBinder<Sublayer3, PolicyContainer<PFeedbackOutput>>,
                                                KVBinder<Sublayer4, PolicyContainer<PFeedbackOutput>>,
                                                KVBinder<Sublayer5, PolicyContainer<PFeedbackOutput>>>>);


        using Sublayer2Left = decltype(declval<Input1Type>() + declval<Input2Type>());
        using Sublayer2Output = decltype(declval<Sublayer2Left>() * declval<Input1Type>());
        using Sublayer3Output = decltype(Sigmoid(declval<Sublayer2Output>()));
        using Sublayer4Output = decltype(Tanh(declval<Sublayer2Output>()));

        using Layer1Final = AddLayer<LayerIOMap<LayerKV<LeftOperand, Input1Type>, LayerKV<RightOperand, Input2Type>>,
                                     PolicyContainer<PNoUpdate, PFeedbackOutput>>;
        using Layer2Final = MultiplyLayer<LayerIOMap<LayerKV<RightOperand, Input1Type>, LayerKV<LeftOperand, Sublayer2Left>>,
                                          PolicyContainer<PUpdate, PFeedbackOutput>>;
        using Layer3Final = SigmoidLayer<LayerIOMap<LayerKV<LayerInput, Sublayer2Output>>,
                                         PolicyContainer<PFeedbackOutput>>;
        using Layer4Final = TanhLayer<LayerIOMap<LayerKV<LayerInput, Sublayer2Output>>,
                                      PolicyContainer<PFeedbackOutput>>;
        using Layer5Final = AddLayer<LayerIOMap<LayerKV<LeftOperand, Sublayer3Output>, LayerKV<RightOperand, Sublayer4Output>>,
                                     PolicyContainer<PFeedbackOutput>>;
        static_assert(std::is_same_v<Check::type, std::tuple<Layer1Final, Layer2Final, Layer3Final, Layer4Final, Layer5Final>>);
        cout << "done" << endl;
    }
    
    void test_compose_kernel8()
    {
        cout << "Test compose kernel case 8...\t";
        using namespace MetaNN::NSComposeKernel;
        using namespace MetaNN::ContMetaFun::Helper;
        
        using CheckInputs = LayerIOMap<LayerKV<Input1, NullParameter>,
                                       LayerKV<Input2, NullParameter>>;
        using CheckPolicyContainer = PolicyContainer<SubPolicyContainer<Sublayer1, PNoUpdate, PFeedbackNoOutput>,
                                                     SubPolicyContainer<Sublayer2, PFeedbackNoOutput>,
                                                     SubPolicyContainer<Sublayer5, PFeedbackNoOutput>>;
        using CheckSublayerCont = ClauseSeq<Sublayer1, Sublayer2, Sublayer3, Sublayer4, Sublayer5>;
        using CheckSublayerClause = ClauseSeq<Sublayer<Sublayer1, AddLayer>,
                                              Sublayer<Sublayer2, MultiplyLayer>,
                                              Sublayer<Sublayer3, SigmoidLayer>,
                                              Sublayer<Sublayer4, TanhLayer>,
                                              Sublayer<Sublayer5, AddLayer>>;
        using CheckInConnects = ClauseSeq<InConnect<Input1, Sublayer1, LeftOperand>,
                                          InConnect<Input2, Sublayer1, RightOperand>,
                                          InConnect<Input1, Sublayer2, RightOperand>>;
        using CheckInterConnects = ClauseSeq<InternalConnect<Sublayer1, LayerOutput, Sublayer2, LeftOperand>,
                                             InternalConnect<Sublayer2, LayerOutput, Sublayer3, LayerInput>,
                                             InternalConnect<Sublayer2, LayerOutput, Sublayer4, LayerInput>,
                                             InternalConnect<Sublayer3, LayerOutput, Sublayer5, LeftOperand>,
                                             InternalConnect<Sublayer4, LayerOutput, Sublayer5, RightOperand>>;
        using CheckOutConnects = ClauseSeq<OutConnect<Sublayer5, LayerOutput, Output1>>;

        using Check = SublayerInstantiation_<CheckInputs, CheckPolicyContainer,
                                             CheckSublayerCont, CheckSublayerClause, CheckInConnects, CheckInterConnects, CheckOutConnects>;

        static_assert(std::is_same_v<Check::SublayerPolicyFinal,
                                     std::tuple<KVBinder<Sublayer1, PolicyContainer<PNoUpdate, PFeedbackNoOutput>>,
                                                KVBinder<Sublayer2, PolicyContainer<PFeedbackNoOutput>>,
                                                KVBinder<Sublayer3, PolicyContainer<>>,
                                                KVBinder<Sublayer4, PolicyContainer<>>,
                                                KVBinder<Sublayer5, PolicyContainer<PFeedbackNoOutput>>>>);


        using Layer1Final = AddLayer<NullParameter, PolicyContainer<PNoUpdate, PFeedbackNoOutput>>;
        using Layer2Final = MultiplyLayer<NullParameter, PolicyContainer<PFeedbackNoOutput>>;
        using Layer3Final = SigmoidLayer<NullParameter, PolicyContainer<>>;
        using Layer4Final = TanhLayer<NullParameter, PolicyContainer<>>;
        using Layer5Final = AddLayer<NullParameter, PolicyContainer<PFeedbackNoOutput>>;

        static_assert(std::is_same_v<Check::type, std::tuple<Layer1Final, Layer2Final, Layer3Final, Layer4Final, Layer5Final>>);
        cout << "done" << endl;
    }
    
    void test_compose_kernel9()
    {
        cout << "Test compose kernel case 9...\t";
        
        using CT = ComposeTopology<Sublayer<Sublayer1, AddLayer>,
                                   Sublayer<Sublayer2, MultiplyLayer>,
                                   Sublayer<Sublayer3, SigmoidLayer>,
                                   Sublayer<Sublayer4, TanhLayer>,
                                   Sublayer<Sublayer5, AddLayer>,
                                   InConnect<Input1, Sublayer1, LeftOperand>,
                                   InConnect<Input2, Sublayer1, RightOperand>,
                                   InConnect<Input1, Sublayer2, RightOperand>,
                                   InternalConnect<Sublayer1, LayerOutput, Sublayer2, LeftOperand>,
                                   InternalConnect<Sublayer2, LayerOutput, Sublayer3, LayerInput>,
                                   InternalConnect<Sublayer2, LayerOutput, Sublayer4, LayerInput>,
                                   InternalConnect<Sublayer3, LayerOutput, Sublayer5, LeftOperand>,
                                   InternalConnect<Sublayer4, LayerOutput, Sublayer5, RightOperand>,
                                   OutConnect<Sublayer5, LayerOutput, Output1>>;
        
        using Input1Type = Matrix<CheckElement, CheckDevice>;
        using Input2Type = TrivalMatrix<Scalar<CheckElement, CheckDevice>>;
        using CheckInputs = LayerIOMap<LayerKV<Input1, Input1Type>,
                                       LayerKV<Input2, Input2Type>>;
        using CheckPolicyContainer = PolicyContainer<PFeedbackOutput,
                                                     SubPolicyContainer<Sublayer1, PNoUpdate, PFeedbackNoOutput>,
                                                     SubPolicyContainer<Sublayer2, PUpdate, PFeedbackNoOutput>,
                                                     SubPolicyContainer<Sublayer5, PFeedbackNoOutput>>;

        using Sublayer2Left = decltype(declval<Input1Type>() + declval<Input2Type>());
        using Sublayer2Output = decltype(declval<Sublayer2Left>() * declval<Input1Type>());
        using Sublayer3Output = decltype(Sigmoid(declval<Sublayer2Output>()));
        using Sublayer4Output = decltype(Tanh(declval<Sublayer2Output>()));
        
        using Layer1Final = AddLayer<LayerIOMap<LayerKV<LeftOperand, Input1Type>, LayerKV<RightOperand, Input2Type>>,
                                     PolicyContainer<PNoUpdate, PFeedbackOutput>>;
        using Layer2Final = MultiplyLayer<LayerIOMap<LayerKV<RightOperand, Input1Type>, LayerKV<LeftOperand, Sublayer2Left>>,
                                          PolicyContainer<PUpdate, PFeedbackOutput>>;
        using Layer3Final = SigmoidLayer<LayerIOMap<LayerKV<LayerInput, Sublayer2Output>>,
                                         PolicyContainer<PFeedbackOutput>>;
        using Layer4Final = TanhLayer<LayerIOMap<LayerKV<LayerInput, Sublayer2Output>>,
                                      PolicyContainer<PFeedbackOutput>>;
        using Layer5Final = AddLayer<LayerIOMap<LayerKV<LeftOperand, Sublayer3Output>, LayerKV<RightOperand, Sublayer4Output>>,
                                     PolicyContainer<PFeedbackOutput>>;

        using check = CT::Instances<CheckInputs, CheckPolicyContainer>;
        static_assert(std::is_same_v<check, std::tuple<Layer1Final, Layer2Final, Layer3Final, Layer4Final, Layer5Final>>);
        cout << "done" << endl;
    }
    
    void test_compose_kernel10()
    {
        cout << "Test compose kernel case 10...\t";
        using CT = ComposeTopology<Sublayer<Sublayer1, AddLayer>,
                                   Sublayer<Sublayer2, MultiplyLayer>,
                                   Sublayer<Sublayer3, SigmoidLayer>,
                                   Sublayer<Sublayer4, TanhLayer>,
                                   Sublayer<Sublayer5, AddLayer>,
                                   InConnect<Input1, Sublayer1, LeftOperand>,
                                   InConnect<Input2, Sublayer1, RightOperand>,
                                   InConnect<Input1, Sublayer2, RightOperand>,
                                   InternalConnect<Sublayer1, LayerOutput, Sublayer2, LeftOperand>,
                                   InternalConnect<Sublayer2, LayerOutput, Sublayer3, LayerInput>,
                                   InternalConnect<Sublayer2, LayerOutput, Sublayer4, LayerInput>,
                                   InternalConnect<Sublayer3, LayerOutput, Sublayer5, LeftOperand>,
                                   InternalConnect<Sublayer4, LayerOutput, Sublayer5, RightOperand>,
                                   OutConnect<Sublayer5, LayerOutput, Output1>>;
        
        using Input1Type = Matrix<CheckElement, CheckDevice>;
        using Input2Type = TrivalMatrix<Scalar<CheckElement, CheckDevice>>;
        using CheckInputs = LayerIOMap<LayerKV<Input1, Input1Type>,
                                       LayerKV<Input2, Input2Type>>;
        using CheckPolicyContainer = PolicyContainer<PFeedbackOutput,
                                                     SubPolicyContainer<Sublayer1, PNoUpdate, PFeedbackNoOutput>,
                                                     SubPolicyContainer<Sublayer2, PUpdate, PFeedbackNoOutput>,
                                                     SubPolicyContainer<Sublayer5, PFeedbackNoOutput>>;
        using Check = ComposeKernel<LayerPortSet<Input1, Input2>, LayerPortSet<Output1>, CheckInputs, CheckPolicyContainer, CT>;
        
        auto sublayerArray = Check::CreateSublayers();
        using ArrayType = decltype(sublayerArray);
        
        using Sublayer2Left = decltype(declval<Input1Type>() + declval<Input2Type>());
        using Sublayer2Output = decltype(declval<Sublayer2Left>() * declval<Input1Type>());
        using Sublayer3Output = decltype(Sigmoid(declval<Sublayer2Output>()));
        using Sublayer4Output = decltype(Tanh(declval<Sublayer2Output>()));
        
        using Layer1Final = AddLayer<LayerIOMap<LayerKV<LeftOperand, Input1Type>, LayerKV<RightOperand, Input2Type>>,
                                     PolicyContainer<PNoUpdate, PFeedbackOutput>>;
        using Layer2Final = MultiplyLayer<LayerIOMap<LayerKV<RightOperand, Input1Type>, LayerKV<LeftOperand, Sublayer2Left>>,
                                          PolicyContainer<PUpdate, PFeedbackOutput>>;
        using Layer3Final = SigmoidLayer<LayerIOMap<LayerKV<LayerInput, Sublayer2Output>>,
                                         PolicyContainer<PFeedbackOutput>>;
        using Layer4Final = TanhLayer<LayerIOMap<LayerKV<LayerInput, Sublayer2Output>>,
                                      PolicyContainer<PFeedbackOutput>>;
        using Layer5Final = AddLayer<LayerIOMap<LayerKV<LeftOperand, Sublayer3Output>, LayerKV<RightOperand, Sublayer4Output>>,
                                     PolicyContainer<PFeedbackOutput>>;

        static_assert(std::is_same_v<ArrayType,
                                     NSComposeKernel::SublayerArrayMaker<NSComposeKernel::ClauseSeq<Sublayer1, Sublayer2, Sublayer3, Sublayer4, Sublayer5>, 
                                                                         std::tuple<Layer1Final, Layer2Final, Layer3Final, Layer4Final, Layer5Final>>>);
        static_assert(std::is_same_v<typename ArrayType::SublayerArray,
                                     std::tuple<std::shared_ptr<Layer1Final>, std::shared_ptr<Layer2Final>,
                                                std::shared_ptr<Layer3Final>, std::shared_ptr<Layer4Final>,
                                                std::shared_ptr<Layer5Final>>>);
        static_assert(Check::IsFeedbackOutput);
        static_assert(!Check::IsUpdate);    // Note: although Layer2Final has update as its policy, the layer itself is not updated.
        
        Check vCheck(Check::CreateSublayers().Set<Sublayer1>("MySublayer1")
                                             .Set<Sublayer2>("MySublayer2")
                                             .Set<Sublayer3>("MySublayer3")
                                             .Set<Sublayer4>("MySublayer4")
                                             .Set<Sublayer5>("MySublayer5"));
        cout << "done" << endl;
    }
}

namespace Test::Layer::Compose
{
    void test_compose_kenrel()
    {
        test_compose_kernel1();
        test_compose_kernel2();
        test_compose_kernel3();
        test_compose_kernel4();
        test_compose_kernel5();
        test_compose_kernel6();
        test_compose_kernel7();
        test_compose_kernel8();
        test_compose_kernel9();
        test_compose_kernel10();
    }
}