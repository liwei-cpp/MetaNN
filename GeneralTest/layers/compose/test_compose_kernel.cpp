#include <iostream>
#include <cassert>
#include <set>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
struct Tag1; struct Tag2; struct Tag3; struct Tag4; struct Tag5; struct Tag6;
struct Input1; struct Input2; struct Input3;
struct Output1; struct Output2;

void test_compose_kernel1()
{
    cout << "Test compose kernel case 1...\t";

    using namespace MetaNN;
    using namespace MetaNN::NSComposeKernel;
    using namespace MetaNN::ContMetaFun;

    using check1 = SeparateClauses_<Sublayer<Tag1, AddLayer>,
                                    Sublayer<Tag2, ElementMulLayer>,
                                    Sublayer<Tag3, BiasLayer>,
                                    Sublayer<Tag4, TanhLayer>,
                                    Sublayer<Tag5, AddLayer>,
                                    InConnect<Input1, Tag1, AddLayerIn1>,
                                    InConnect<Input2, Tag1, AddLayerIn2>,
                                    InConnect<Input1, Tag2, ElementMulLayerIn2>,
                                    InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                    InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                    InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                    InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                    InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>,
                                    OutConnect<Tag5, LayerIO, Output1>>;
    static_assert(std::is_same_v<check1::SublayerRes,
                                 ClauseSeq<Sublayer<Tag1, AddLayer>,
                                           Sublayer<Tag2, ElementMulLayer>,
                                           Sublayer<Tag3, BiasLayer>,
                                           Sublayer<Tag4, TanhLayer>,
                                           Sublayer<Tag5, AddLayer>>>);
                                         
    static_assert(std::is_same_v<check1::InConnectRes,
                                 ClauseSeq<InConnect<Input1, Tag1, AddLayerIn1>,
                                           InConnect<Input2, Tag1, AddLayerIn2>,
                                           InConnect<Input1, Tag2, ElementMulLayerIn2>>>);

    static_assert(std::is_same_v<check1::OutConnectRes,
                                 ClauseSeq<OutConnect<Tag5, LayerIO, Output1>>>);
                                         
    static_assert(std::is_same_v<ClauseRefine::InternalFMap<check1::InterConnectRes>,
                                 ClauseSeq<Helper::Pair<Tag1, InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>>,
                                           Helper::Pair<Tag2, InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                              InternalConnect<Tag2, LayerIO, Tag4, LayerIO>>,
                                           Helper::Pair<Tag3, InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>>,
                                           Helper::Pair<Tag4, InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>>>);
                                           
    static_assert(std::is_same_v<ClauseRefine::InternalBMap<check1::InterConnectRes>,
                                 ClauseSeq<Helper::Pair<Tag2, InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>>,
                                           Helper::Pair<Tag3, InternalConnect<Tag2, LayerIO, Tag3, LayerIO>>,
                                           Helper::Pair<Tag4, InternalConnect<Tag2, LayerIO, Tag4, LayerIO>>,
                                           Helper::Pair<Tag5, InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                              InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>>>);
                                                              
    static_assert(std::is_same_v<ClauseRefine::InternalInLayerSet<check1::InterConnectRes>, ClauseSeq<Tag2, Tag3, Tag4, Tag5>>);
    static_assert(std::is_same_v<ClauseRefine::InternalLayerSet<check1::InterConnectRes>, ClauseSeq<Tag2, Tag3, Tag4, Tag5, Tag1>>);
    static_assert(std::is_same_v<ClauseRefine::InputLayerSet<check1::InConnectRes>, ClauseSeq<Tag1, Tag2>>);
    static_assert(std::is_same_v<ClauseRefine::OutputLayerSet<check1::OutConnectRes>, ClauseSeq<Tag5>>);
    
    using check2 = SeparateClauses_<Sublayer<Tag1, AddLayer>,
                                    InConnect<Input1, Tag1, AddLayerIn1>,
                                    InConnect<Input2, Tag1, AddLayerIn2>,
                                    Sublayer<Tag2, ElementMulLayer>,
                                    InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                    InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                    Sublayer<Tag3, BiasLayer>,
                                    InConnect<Input1, Tag2, ElementMulLayerIn2>,
                                    InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                    Sublayer<Tag4, TanhLayer>,
                                    InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                    OutConnect<Tag5, LayerIO, Output1>,
                                    Sublayer<Tag5, AddLayer>,
                                    InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>;

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
    using Check1 = ClauseRefine::SublayerNameSet<ClauseSeq<Sublayer<Tag1, AddLayer>,
                                                 Sublayer<Tag2, ElementMulLayer>,
                                                 Sublayer<Tag3, BiasLayer>>>;
    static_assert(std::is_same_v<Check1, ClauseSeq<Tag1, Tag2, Tag3>>);
    
    using Check2 = ClauseRefine::SublayerNameSet<ClauseSeq<Sublayer<Tag1, AddLayer>,
                                                 Sublayer<Tag2, ElementMulLayer>,
                                                 Sublayer<Tag2, BiasLayer>>>;
    static_assert(ArraySize<Check2> < 3);
    cout << "done" << endl;
}

void test_compose_kernel3()
{
    cout << "Test compose kernel case 3...\t";

    using namespace MetaNN::NSComposeKernel;
    using Check1 = ClauseRefine::InternalInNamePortSet<ClauseSeq<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                                 InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                                 InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                                                 InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                                 InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>>;
    static_assert(ArraySize<Check1> == 5);
    
    using Check2 = ClauseRefine::InternalInNamePortSet<ClauseSeq<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                                 InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                                 InternalConnect<Tag5, LayerIO, Tag3, LayerIO>,
                                                                 InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                                 InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>>;
    static_assert(ArraySize<Check2> < 5);
    cout << "done" << endl;
}

void test_compose_kernel4()
{
    cout << "Test compose kernel case 4...\t";

    using namespace MetaNN::NSComposeKernel;
    using Check1 = ClauseRefine::InputNamePortSet<ClauseSeq<InConnect<Input1, Tag1, AddLayerIn1>,
                                                            InConnect<Input2, Tag1, AddLayerIn2>>>;
    static_assert(ArraySize<Check1> == 2);

    using Check2 = ClauseRefine::InputNamePortSet<ClauseSeq<InConnect<Input1, Tag1, AddLayerIn1>,
                                                            InConnect<Input2, Tag1, AddLayerIn1>>>;
    static_assert(ArraySize<Check2> < 2);
    
    using Check3 = ClauseRefine::InputNamePortSet<ClauseSeq<InConnect<Input1, Tag1, AddLayerIn1>,
                                                            InConnect<Input1, Tag1, AddLayerIn2>>>;
    static_assert(ArraySize<Check3> == 2);
    
    cout << "done" << endl;
}

void test_compose_kernel5()
{
    cout << "Test compose kernel case 5...\t";

    using namespace MetaNN::NSComposeKernel;
    using Check1 = ClauseRefine::OutputPortSet<ClauseSeq<OutConnect<Tag5, LayerIO, Output1>>>;
    static_assert(ArraySize<Check1> == 1);
    
    using Check2 = ClauseRefine::OutputPortSet<ClauseSeq<OutConnect<Tag5, LayerIO, Output1>,
                                                         OutConnect<Tag5, LayerIO, Output2>>>;
    static_assert(ArraySize<Check2> == 2);
    
    using Check3 = ClauseRefine::OutputPortSet<ClauseSeq<OutConnect<Tag5, LayerIO, Output1>,
                                                         OutConnect<Tag5, LayerIO, Output1>>>;
    static_assert(ArraySize<Check3> < 2);
    cout << "done" << endl;
}

void test_compose_kernel6()
{
    cout << "Test compose kernel case 6...\t";
    using namespace MetaNN::NSComposeKernel;

    constexpr bool check1 = InternalTagInSublayer<ClauseSeq<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                            InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                            InternalConnect<Tag5, LayerIO, Tag3, LayerIO>,
                                                            InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                            InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                                  std::tuple<Tag1, Tag2, Tag3, Tag4, Tag5>>;
    static_assert(check1, "Check Error");
    
    constexpr bool check2 = InternalTagInSublayer<ClauseSeq<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                            InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                            InternalConnect<Tag5, LayerIO, Tag3, LayerIO>,
                                                            InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                            InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                                  std::tuple<Tag1, Tag2, Tag3, Tag4>>;
    static_assert(!check2, "Check Error");
    
    constexpr bool check3 = InternalTagInSublayer<ClauseSeq<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                            InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                            InternalConnect<Tag5, LayerIO, Tag3, LayerIO>>,
                                                  std::tuple<Tag1, Tag2, Tag3, Tag4, Tag5>>;
    static_assert(check3, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel7()
{
    cout << "Test compose kernel case 7...\t";
    using namespace MetaNN::NSComposeKernel;

    constexpr bool check1 = InputTagInSubLayer<ClauseSeq<InConnect<Input1, Tag1, AddLayerIn1>,
                                                         InConnect<Input2, Tag1, AddLayerIn2>,
                                                         InConnect<Input1, Tag2, ElementMulLayerIn2>>,
                                               std::tuple<Tag1, Tag2, Tag3, Tag4, Tag5>>;
    static_assert(check1, "Check Error");
    
    constexpr bool check2 = InputTagInSubLayer<ClauseSeq<InConnect<Input1, Tag1, AddLayerIn1>,
                                                         InConnect<Input2, Tag1, AddLayerIn2>,
                                                         InConnect<Input1, Tag2, ElementMulLayerIn2>>,
                                               std::tuple<Tag1, Tag3, Tag4, Tag5>>;
    static_assert(!check2, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel8()
{
    cout << "Test compose kernel case 8...\t";
    using namespace MetaNN::NSComposeKernel;

    constexpr bool check1 = OutputTagInSubLayer<ClauseSeq<OutConnect<Tag5, LayerIO, Output1>>,
                                                std::tuple<Tag1, Tag3, Tag4, Tag5>>;
    static_assert(check1, "Check Error");
    
    constexpr bool check2 = OutputTagInSubLayer<ClauseSeq<OutConnect<Tag5, LayerIO, Output1>>,
                                                std::tuple<Tag1, Tag3, Tag4>>;
    static_assert(!check2, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel9()
{
    cout << "Test compose kernel case 9...\t";
    using namespace MetaNN::NSComposeKernel;
    constexpr bool check1 = SublayerTagInOtherArrays<ClauseSeq<Tag1, Tag2, Tag3, Tag4, Tag5>,
                                                     ClauseSeq<Tag1, Tag2, Tag3, Tag4>,
                                                     ClauseSeq<Tag1, Tag2>,
                                                     ClauseSeq<Tag5>>;
    static_assert(check1);
    
    constexpr bool check2 = SublayerTagInOtherArrays<ClauseSeq<Tag1, Tag2, Tag3, Tag4, Tag5, Tag6>,
                                                     ClauseSeq<Tag1, Tag2, Tag3, Tag4>,
                                                     ClauseSeq<Tag1, Tag2>,
                                                     ClauseSeq<Tag5>>;
    static_assert(!check2);
    cout << "done" << endl;
}

void test_compose_kernel10()
{
    cout << "Test compose kernel case 10...\t";
    using namespace MetaNN::NSComposeKernel;

    constexpr bool check1 = UsefulInternalPostLayer<ClauseSeq<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                              InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                              InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                                              InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                              InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                                    ClauseSeq<Tag5>>;
    static_assert(check1, "Check Error");

    // Error: Tag2 is useless
    constexpr bool check2 = UsefulInternalPostLayer<ClauseSeq<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                              InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                              InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                                    ClauseSeq<Tag5>>;
    static_assert(!check2, "Check Error");
    
    // Error: Tag5 is useless
    constexpr bool check3 = UsefulInternalPostLayer<ClauseSeq<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                              InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                              InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                                              InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                              InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                                    ClauseSeq<>>;
    static_assert(!check3, "Check Error");
    
    constexpr bool check4 = UsefulInternalPostLayer<ClauseSeq<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                              InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                              InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                                    ClauseSeq<Tag5, Tag2>>;
    static_assert(check4, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel11()
{
    cout << "Test compose kernel case 11...\t";
    using namespace MetaNN::NSComposeKernel;

    constexpr bool check1 = UsefulInputLayer<ClauseSeq<InConnect<Input1, Tag1, AddLayerIn1>,
                                                       InConnect<Input2, Tag1, AddLayerIn2>,
                                                       InConnect<Input1, Tag2, ElementMulLayerIn2>>,
                                             ClauseSeq<Tag1, Tag2, Tag3, Tag4>,
                                             ClauseSeq<Tag5>>;
    static_assert(check1, "Check Error");
    
    constexpr bool check2 = UsefulInputLayer<ClauseSeq<InConnect<Input1, Tag1, AddLayerIn1>,
                                                       InConnect<Input2, Tag1, AddLayerIn2>,
                                                       InConnect<Input1, Tag5, ElementMulLayerIn2>>,
                                             ClauseSeq<Tag1, Tag2, Tag3, Tag4>,
                                             ClauseSeq<Tag5>>;
    static_assert(check2, "Check Error");
    
    // Error: Tag1 is neither in InterConnect nor in OutConnect
    constexpr bool check3 = UsefulInputLayer<ClauseSeq<InConnect<Input1, Tag1, AddLayerIn1>,
                                                       InConnect<Input2, Tag1, AddLayerIn2>,
                                                       InConnect<Input1, Tag5, ElementMulLayerIn2>>,
                                             ClauseSeq<Tag2, Tag3, Tag4>,
                                             ClauseSeq<Tag5>>;
    static_assert(!check3, "Check Error");
    
    constexpr bool check4 = UsefulInputLayer<ClauseSeq<InConnect<Input1, Tag1, AddLayerIn1>,
                                                       InConnect<Input2, Tag1, AddLayerIn2>,
                                                       InConnect<Input1, Tag5, ElementMulLayerIn2>>,
                                             ClauseSeq<Tag1, Tag2, Tag3, Tag4>,
                                             ClauseSeq<>>;
    static_assert(!check4, "Check Error");
    
    cout << "done" << endl;
}

void test_compose_kernel12()
{
    cout << "Test compose kernel case 12...\t";
    using namespace MetaNN::NSComposeKernel;

    using InterConnects = ClauseSeq<InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                    InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>,
                                    InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                    InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                    InternalConnect<Tag2, LayerIO, Tag4, LayerIO>>;

    using check1 = TopologicalOrdering_<ClauseSeq<Sublayer<Tag3, BiasLayer>,
                                                  Sublayer<Tag2, ElementMulLayer>,
                                                  Sublayer<Tag1, AddLayer>,
                                                  Sublayer<Tag4, TanhLayer>,
                                                  Sublayer<Tag6, AddLayer>,
                                                  Sublayer<Tag5, AddLayer>>,
                                        InterConnects>::type;
    using comp1 = ClauseSeq<Sublayer<Tag6, AddLayer>,
                            Sublayer<Tag1, AddLayer>,
                            Sublayer<Tag2, ElementMulLayer>,
                            Sublayer<Tag4, TanhLayer>,
                            Sublayer<Tag3, BiasLayer>,
                            Sublayer<Tag5, AddLayer>>;
                            
    static_assert(std::is_same<check1, comp1>::value, "Check Error");

    using Policy1 = PolicyContainer<PFeedbackOutput>;
    using Instantiation1 = SublayerInstantiation<Policy1, check1, InterConnects>::type;
    using InstantiationComp1 = std::tuple<InstantiatedSublayer<Tag6, AddLayer<PolicyContainer<PFeedbackOutput>>>,
                                          InstantiatedSublayer<Tag1, AddLayer<PolicyContainer<PFeedbackOutput>>>,
                                          InstantiatedSublayer<Tag2, ElementMulLayer<PolicyContainer<PFeedbackOutput>>>,
                                          InstantiatedSublayer<Tag4, TanhLayer<PolicyContainer<PFeedbackOutput>>>,
                                          InstantiatedSublayer<Tag3, BiasLayer<PolicyContainer<PFeedbackOutput>>>,
                                          InstantiatedSublayer<Tag5, AddLayer<PolicyContainer<PFeedbackOutput>>>>;
    static_assert(std::is_same<Instantiation1, InstantiationComp1>::value, "Check Error");

    using Policy2 = PolicyContainer<PTanhAction, SubPolicyContainer<Tag3, PBatchMode>>;
    using Instantiation2 = SublayerInstantiation<Policy2, check1, InterConnects>::type;
    using InstantiationComp2 = std::tuple<InstantiatedSublayer<Tag6, AddLayer<PolicyContainer<PTanhAction>>>,
                                          InstantiatedSublayer<Tag1, AddLayer<PolicyContainer<PTanhAction>>>,
                                          InstantiatedSublayer<Tag2, ElementMulLayer<PolicyContainer<PTanhAction>>>,
                                          InstantiatedSublayer<Tag4, TanhLayer<PolicyContainer<PTanhAction>>>,
                                          InstantiatedSublayer<Tag3, BiasLayer<PolicyContainer<PBatchMode, PTanhAction>>>,
                                          InstantiatedSublayer<Tag5, AddLayer<PolicyContainer<PTanhAction>>>>;
    static_assert(std::is_same<Instantiation2, InstantiationComp2>::value, "Check Error");

    using Policy3 = PolicyContainer<PTanhAction, SubPolicyContainer<Tag2, PUpdate>>;
    using Instantiation3 = SublayerInstantiation<Policy3, check1, InterConnects>::type;
    using InstantiationComp3 = std::tuple<InstantiatedSublayer<Tag6, AddLayer<PolicyContainer<PTanhAction>>>,
                                          InstantiatedSublayer<Tag1, AddLayer<PolicyContainer<PTanhAction>>>,
                                          InstantiatedSublayer<Tag2, ElementMulLayer<PolicyContainer<PUpdate, PTanhAction>>>,
                                          InstantiatedSublayer<Tag4, TanhLayer<PolicyContainer<PTanhAction, PFeedbackOutput>>>,
                                          InstantiatedSublayer<Tag3, BiasLayer<PolicyContainer<PTanhAction, PFeedbackOutput>>>,
                                          InstantiatedSublayer<Tag5, AddLayer<PolicyContainer<PTanhAction, PFeedbackOutput>>>>;
    static_assert(std::is_same<Instantiation3, InstantiationComp3>::value, "Check Error");

    using Policy4 = PolicyContainer<PTanhAction, SubPolicyContainer<Tag3, PUpdate>>;
    using Instantiation4 = SublayerInstantiation<Policy4, check1, InterConnects>::type;
    using InstantiationComp4 = std::tuple<InstantiatedSublayer<Tag6, AddLayer<PolicyContainer<PTanhAction>>>,
                                          InstantiatedSublayer<Tag1, AddLayer<PolicyContainer<PTanhAction>>>,
                                          InstantiatedSublayer<Tag2, ElementMulLayer<PolicyContainer<PTanhAction>>>,
                                          InstantiatedSublayer<Tag4, TanhLayer<PolicyContainer<PTanhAction>>>,
                                          InstantiatedSublayer<Tag3, BiasLayer<PolicyContainer<PUpdate, PTanhAction>>>,
                                          InstantiatedSublayer<Tag5, AddLayer<PolicyContainer<PTanhAction, PFeedbackOutput>>>>;
    static_assert(std::is_same<Instantiation4, InstantiationComp4>::value, "Check Error");
    cout << "done" << endl;
}
}

void test_compose_kernel()
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
    test_compose_kernel11();
    test_compose_kernel12();
}
