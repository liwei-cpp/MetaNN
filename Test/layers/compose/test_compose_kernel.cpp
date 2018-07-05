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

    using namespace MetaNN::NSComposeKernel;

    using check1 = SeparateParameters_<SubLayer<Tag1, AddLayer>,
                                      SubLayer<Tag2, ElementMulLayer>,
                                      SubLayer<Tag3, BiasLayer>,
                                      SubLayer<Tag4, TanhLayer>,
                                      SubLayer<Tag5, AddLayer>,
                                      InConnect<Input1, Tag1, AddLayerIn1>,
                                      InConnect<Input2, Tag1, AddLayerIn2>,
                                      InConnect<Input1, Tag2, ElementMulLayerIn2>,
                                      InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                      InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                      InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                      InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                      InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>,
                                      OutConnect<Tag5, LayerIO, Output1>>;
    static_assert(std::is_same<check1::SubLayerRes,
                               SubLayerContainer<SubLayer<Tag1, AddLayer>,
                                                 SubLayer<Tag2, ElementMulLayer>,
                                                 SubLayer<Tag3, BiasLayer>,
                                                 SubLayer<Tag4, TanhLayer>,
                                                 SubLayer<Tag5, AddLayer>>>::value, "Check Error");
    static_assert(std::is_same<check1::InterConnectRes,
                               InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                     InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                     InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                                     InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                     InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>>::value, "Check Error");
    static_assert(std::is_same<check1::InConnectRes,
                               InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
                                                  InConnect<Input2, Tag1, AddLayerIn2>,
                                                  InConnect<Input1, Tag2, ElementMulLayerIn2>>>::value, "Check Error");

    static_assert(std::is_same<check1::OutConnectRes,
                               OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>>::value, "Check Error");

    using check2 = SeparateParameters_<SubLayer<Tag1, AddLayer>,
                                      InConnect<Input1, Tag1, AddLayerIn1>,
                                      InConnect<Input2, Tag1, AddLayerIn2>,
                                      SubLayer<Tag2, ElementMulLayer>,
                                      InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                      InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                      SubLayer<Tag3, BiasLayer>,
                                      InConnect<Input1, Tag2, ElementMulLayerIn2>,
                                      InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                      SubLayer<Tag4, TanhLayer>,
                                      InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                      OutConnect<Tag5, LayerIO, Output1>,
                                      SubLayer<Tag5, AddLayer>,
                                      InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>;

    static_assert(std::is_same<check2::SubLayerRes, check1::SubLayerRes>::value, "Check Error");
    static_assert(std::is_same<check2::InterConnectRes, check1::InterConnectRes>::value, "Check Error");
    static_assert(std::is_same<check2::InConnectRes, check1::InConnectRes>::value, "Check Error");
    static_assert(std::is_same<check2::OutConnectRes, check1::OutConnectRes>::value, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel2()
{
    cout << "Test compose kernel case 2...\t";

    using namespace MetaNN::NSComposeKernel;
    static_assert(TagExist<Tag1, Tag5, Tag4, Tag1, Tag3>::value, "Check Error");
    static_assert(!TagExist<Tag1, Tag5, Tag4, Tag3>::value, "Check Error");

    static_assert(TagExistInLayerComps<Tag2, SubLayer<Tag1, AddLayer>,
                                             SubLayer<Tag2, ElementMulLayer>,
                                             SubLayer<Tag3, BiasLayer>>::value, "Check Error");
    static_assert(!TagExistInLayerComps<Tag4, SubLayer<Tag1, AddLayer>,
                                              SubLayer<Tag2, ElementMulLayer>,
                                              SubLayer<Tag3, BiasLayer>>::value, "Check Error");

    static_assert(TagExistInLayerComps<Tag2, InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                             InternalConnect<Tag2, LayerIO, Tag4, LayerIO>>::value, "Check Error");
    static_assert(TagExistInLayerComps<Tag3, InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                             InternalConnect<Tag2, LayerIO, Tag4, LayerIO>>::value, "Check Error");
    static_assert(TagExistInLayerComps<Tag4, InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                             InternalConnect<Tag2, LayerIO, Tag4, LayerIO>>::value, "Check Error");
    static_assert(!TagExistInLayerComps<Tag1, InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                              InternalConnect<Tag2, LayerIO, Tag4, LayerIO>>::value, "Check Error");

    static_assert(TagExistInLayerComps<Tag2, InConnect<Input1, Tag1, AddLayerIn1>,
                                             InConnect<Input2, Tag1, AddLayerIn2>,
                                             InConnect<Input1, Tag2, ElementMulLayerIn2>>::value, "Check Error");
    static_assert(!TagExistInLayerComps<Tag3, InConnect<Input1, Tag1, AddLayerIn1>,
                                              InConnect<Input2, Tag1, AddLayerIn2>,
                                              InConnect<Input1, Tag2, ElementMulLayerIn2>>::value, "Check Error");

    static_assert(TagExistInLayerComps<Tag5, OutConnect<Tag5, LayerIO, Output1>>::value, "Check Error");
    static_assert(!TagExistInLayerComps<Tag3, OutConnect<Tag5, LayerIO, Output1>>::value, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel3()
{
    cout << "Test compose kernel case 3...\t";

    using namespace MetaNN::NSComposeKernel;
    static_assert(SublayerCheck<SubLayerContainer<SubLayer<Tag1, AddLayer>,
                                                  SubLayer<Tag2, ElementMulLayer>,
                                                  SubLayer<Tag3, BiasLayer>>>::IsUnique, "Check Error");
    static_assert(!SublayerCheck<SubLayerContainer<SubLayer<Tag1, AddLayer>,
                                                   SubLayer<Tag2, ElementMulLayer>,
                                                   SubLayer<Tag2, BiasLayer>>>::IsUnique, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel4()
{
    cout << "Test compose kernel case 4...\t";

    using namespace MetaNN::NSComposeKernel;
    using check1 = InternalConnectCheck<InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                              InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                              InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                                              InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                              InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>>;
    static_assert(check1::NoSelfCycle, "Check Error");
    static_assert(check1::UniqueSource, "Check Error");

    using check2 = InternalConnectCheck<InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                              InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                              InternalConnect<Tag2, LayerIO, Tag2, LayerIO>,
                                                              InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                              InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>>;
    static_assert(!check2::NoSelfCycle, "Check Error");
    static_assert(check2::UniqueSource, "Check Error");

    using check3 = InternalConnectCheck<InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                              InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                              InternalConnect<Tag5, LayerIO, Tag3, LayerIO>,
                                                              InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                              InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>>;
    static_assert(check3::NoSelfCycle, "Check Error");
    static_assert(!check3::UniqueSource, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel5()
{
    cout << "Test compose kernel case 5...\t";

    using namespace MetaNN::NSComposeKernel;
    using check1 = InputConnectCheck<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
                                                        InConnect<Input2, Tag1, AddLayerIn2>>>;
    static_assert(check1::UniqueSource, "Check Error");

    using check2 = InputConnectCheck<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
                                                        InConnect<Input2, Tag1, AddLayerIn1>>>;
    static_assert(!check2::UniqueSource, "Check Error");

    using check3 = InputConnectCheck<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
                                                        InConnect<Input1, Tag1, AddLayerIn2>>>;
    static_assert(check3::UniqueSource, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel6()
{
    cout << "Test compose kernel case 6...\t";

    using namespace MetaNN::NSComposeKernel;
    using check1 = OutputConnectCheck<OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>>;
    static_assert(check1::UniqueSource, "Check Error");

    using check2 = OutputConnectCheck<OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>,
                                                          OutConnect<Tag5, LayerIO, Output2>>>;
    static_assert(check2::UniqueSource, "Check Error");

    using check3 = OutputConnectCheck<OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>,
                                                          OutConnect<Tag3, LayerIO, Output1>>>;
    static_assert(!check3::UniqueSource, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel7()
{
    cout << "Test compose kernel case 7...\t";
    using namespace MetaNN::NSComposeKernel;

    using check1 = InternalTagInSublayer<InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                               InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                               InternalConnect<Tag5, LayerIO, Tag3, LayerIO>,
                                                               InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                               InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                          SubLayerContainer<SubLayer<Tag1, AddLayer>,
                                                            SubLayer<Tag2, ElementMulLayer>,
                                                            SubLayer<Tag3, BiasLayer>,
                                                            SubLayer<Tag4, TanhLayer>,
                                                            SubLayer<Tag5, AddLayer>>>;
    static_assert(check1::value, "Check Error");

    using check2 = InternalTagInSublayer<InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                               InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                               InternalConnect<Tag5, LayerIO, Tag3, LayerIO>,
                                                               InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                               InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                         SubLayerContainer<SubLayer<Tag1, AddLayer>,
                                                           SubLayer<Tag2, ElementMulLayer>,
                                                           SubLayer<Tag3, BiasLayer>,
                                                           SubLayer<Tag4, TanhLayer>>>;
    static_assert(!check2::value, "Check Error");

    using check3 = InternalTagInSublayer<InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                               InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                               InternalConnect<Tag5, LayerIO, Tag3, LayerIO>>,
                                         SubLayerContainer<SubLayer<Tag1, AddLayer>,
                                                           SubLayer<Tag2, ElementMulLayer>,
                                                           SubLayer<Tag3, BiasLayer>,
                                                           SubLayer<Tag4, TanhLayer>,
                                                           SubLayer<Tag5, AddLayer>>>;
    static_assert(check3::value, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel8()
{
    cout << "Test compose kernel case 8...\t";
    using namespace MetaNN::NSComposeKernel;

    using check1 = InputTagInSubLayer<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
                                                         InConnect<Input2, Tag1, AddLayerIn2>,
                                                         InConnect<Input1, Tag2, ElementMulLayerIn2>>,
                                      SubLayerContainer<SubLayer<Tag1, AddLayer>,
                                                        SubLayer<Tag2, ElementMulLayer>,
                                                        SubLayer<Tag3, BiasLayer>,
                                                        SubLayer<Tag4, TanhLayer>,
                                                        SubLayer<Tag5, AddLayer>>>;
    static_assert(check1::value, "Check Error");

    using check2 = InputTagInSubLayer<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
                                                         InConnect<Input2, Tag1, AddLayerIn2>,
                                                         InConnect<Input1, Tag2, ElementMulLayerIn2>>,
                                      SubLayerContainer<SubLayer<Tag1, AddLayer>,
                                                        SubLayer<Tag3, BiasLayer>,
                                                        SubLayer<Tag4, TanhLayer>,
                                                        SubLayer<Tag5, AddLayer>>>;
    static_assert(!check2::value, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel9()
{
    cout << "Test compose kernel case 9...\t";
    using namespace MetaNN::NSComposeKernel;

    using check1 = OutputTagInSubLayer<OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>,
                                       SubLayerContainer<SubLayer<Tag1, AddLayer>,
                                                         SubLayer<Tag2, ElementMulLayer>,
                                                         SubLayer<Tag3, BiasLayer>,
                                                         SubLayer<Tag4, TanhLayer>,
                                                         SubLayer<Tag5, AddLayer>>>;
    static_assert(check1::value, "Check Error");

    using check2 = OutputTagInSubLayer<OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>,
                                       SubLayerContainer<SubLayer<Tag1, AddLayer>,
                                                         SubLayer<Tag3, BiasLayer>,
                                                         SubLayer<Tag4, TanhLayer>>>;
    static_assert(!check2::value, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel10()
{
    cout << "Test compose kernel case 10...\t";
    using namespace MetaNN::NSComposeKernel;

    using check1 = SublayerTagInOtherArrays<InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                                  InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                                  InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                                                  InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                                  InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                            InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
                                                               InConnect<Input2, Tag1, AddLayerIn2>,
                                                               InConnect<Input1, Tag2, ElementMulLayerIn2>>,
                                            OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>,
                                            SubLayerContainer<SubLayer<Tag1, AddLayer>,
                                                              SubLayer<Tag2, ElementMulLayer>,
                                                              SubLayer<Tag3, BiasLayer>,
                                                              SubLayer<Tag4, TanhLayer>,
                                                              SubLayer<Tag5, AddLayer>>>;
    static_assert(check1::value, "Check Error");

    using check2 = SublayerTagInOtherArrays<InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                                  InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                                  InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                                                  InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                                  InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                            InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
                                                               InConnect<Input2, Tag1, AddLayerIn2>,
                                                               InConnect<Input1, Tag2, ElementMulLayerIn2>>,
                                            OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>,
                                            SubLayerContainer<SubLayer<Tag1, AddLayer>,
                                                              SubLayer<Tag2, ElementMulLayer>,
                                                              SubLayer<Tag3, BiasLayer>,
                                                              SubLayer<Tag4, TanhLayer>,
                                                              SubLayer<Tag5, AddLayer>,
                                                              SubLayer<Tag6, AddLayer>>>;
    static_assert(!check2::value, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel11()
{
    cout << "Test compose kernel case 11...\t";
    using namespace MetaNN::NSComposeKernel;

    using check1 = TagInInternalOut<Tag2, InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                          InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                          InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                          InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                          InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>;
    static_assert(check1::value, "Check Error");

    using check2 = TagInInternalOut<Tag5, InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                          InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                          InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                          InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                          InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>;
    static_assert(!check2::value, "Check Error");

    using check3 = TagInInternalIn<Tag2, InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                         InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                         InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                         InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                         InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>;
    static_assert(check3::value, "Check Error");

    using check4 = TagInInternalIn<Tag1, InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                         InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                         InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                         InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                         InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>;
    static_assert(!check4::value, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel12()
{
    cout << "Test compose kernel case 12...\t";
    using namespace MetaNN::NSComposeKernel;

    using check1 = UsefulInternalPostLayer<InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                                 InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                                 InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                                                 InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                                 InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                           OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>>;
    static_assert(check1::value, "Check Error");

    // Error: Tag2 is useless
    using check2 = UsefulInternalPostLayer<InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                                 InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                                 InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                           OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>>;
    static_assert(!check2::value, "Check Error");

    // Error: Tag5 is useless
    using check3 = UsefulInternalPostLayer<InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                                 InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                                 InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                                                 InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                                 InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                           OutConnectContainer<>>;
    static_assert(!check3::value, "Check Error");

    using check4 = UsefulInternalPostLayer<InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                                 InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                                 InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                           OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>,
                                                               OutConnect<Tag2, LayerIO, Output2>>>;
    static_assert(check4::value, "Check Error");
    cout << "done" << endl;
}

void test_compose_kernel13()
{
    cout << "Test compose kernel case 13...\t";
    using namespace MetaNN::NSComposeKernel;

    using check1 = UsefulInputLayer<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
                                                       InConnect<Input2, Tag1, AddLayerIn2>,
                                                       InConnect<Input1, Tag2, ElementMulLayerIn2>>,
                                    InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                          InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                          InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                                          InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                          InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                    OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>>;
    static_assert(check1::value, "Check Error");

    using check2 = UsefulInputLayer<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
                                                       InConnect<Input2, Tag1, AddLayerIn2>,
                                                       InConnect<Input1, Tag5, ElementMulLayerIn2>>,
                                    InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                          InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                          InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                                          InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                          InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                    OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>>;
    static_assert(check2::value, "Check Error");

    // Error: Tag1 is neither in InterConnect nor in OutConnect
    using check3 = UsefulInputLayer<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
                                                       InConnect<Input2, Tag1, AddLayerIn2>,
                                                       InConnect<Input1, Tag5, ElementMulLayerIn2>>,
                                    InterConnectContainer<InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                          InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                                          InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                          InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                    OutConnectContainer<OutConnect<Tag5, LayerIO, Output1>>>;
    static_assert(!check3::value, "Check Error");

    using check4 = UsefulInputLayer<InConnectContainer<InConnect<Input1, Tag1, AddLayerIn1>,
                                                       InConnect<Input2, Tag1, AddLayerIn2>,
                                                       InConnect<Input1, Tag5, ElementMulLayerIn2>>,
                                    InterConnectContainer<InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                          InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                          InternalConnect<Tag2, LayerIO, Tag4, LayerIO>,
                                                          InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                          InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>>,
                                    OutConnectContainer<>>;
    static_assert(!check4::value, "Check Error");

    cout << "done" << endl;
}

void test_compose_kernel14()
{
    cout << "Test compose kernel case 14...\t";
    using namespace MetaNN::NSComposeKernel;

    using InterConnects = InterConnectContainer<InternalConnect<Tag2, LayerIO, Tag3, LayerIO>,
                                                InternalConnect<Tag4, LayerIO, Tag5, AddLayerIn2>,
                                                InternalConnect<Tag1, LayerIO, Tag2, ElementMulLayerIn1>,
                                                InternalConnect<Tag3, LayerIO, Tag5, AddLayerIn1>,
                                                InternalConnect<Tag2, LayerIO, Tag4, LayerIO>>;

    using check1 = TopologicalOrdering_<SubLayerContainer<SubLayer<Tag3, BiasLayer>,
                                                          SubLayer<Tag2, ElementMulLayer>,
                                                          SubLayer<Tag1, AddLayer>,
                                                          SubLayer<Tag4, TanhLayer>,
                                                          SubLayer<Tag6, AddLayer>,
                                                          SubLayer<Tag5, AddLayer>>,
                                            InterConnects>::type;
    using comp1 = SubLayerContainer<SubLayer<Tag6, AddLayer>,
                                    SubLayer<Tag1, AddLayer>,
                                    SubLayer<Tag2, ElementMulLayer>,
                                    SubLayer<Tag3, BiasLayer>,
                                    SubLayer<Tag4, TanhLayer>,
                                    SubLayer<Tag5, AddLayer>>;
    static_assert(std::is_same<check1, comp1>::value, "Check Error");

    using Policy1 = PolicyContainer<PFeedbackOutput>;
    using Instantiation1 = SublayerInstantiation<Policy1, check1, InterConnects>::type;
    using InstantiationComp1 = std::tuple<InstantiatedSublayer<Tag6, AddLayer<PolicyContainer<PFeedbackOutput>>>,
                                                             InstantiatedSublayer<Tag1, AddLayer<PolicyContainer<PFeedbackOutput>>>,
                                                             InstantiatedSublayer<Tag2, ElementMulLayer<PolicyContainer<PFeedbackOutput>>>,
                                                             InstantiatedSublayer<Tag3, BiasLayer<PolicyContainer<PFeedbackOutput>>>,
                                                             InstantiatedSublayer<Tag4, TanhLayer<PolicyContainer<PFeedbackOutput>>>,
                                                             InstantiatedSublayer<Tag5, AddLayer<PolicyContainer<PFeedbackOutput>>>>;
    static_assert(std::is_same<Instantiation1, InstantiationComp1>::value, "Check Error");

    using Policy2 = PolicyContainer<PTanhAction, SubPolicyContainer<Tag3, PBatchMode>>;
    using Instantiation2 = SublayerInstantiation<Policy2, check1, InterConnects>::type;
    using InstantiationComp2 = std::tuple<InstantiatedSublayer<Tag6, AddLayer<PolicyContainer<PTanhAction>>>,
                                                             InstantiatedSublayer<Tag1, AddLayer<PolicyContainer<PTanhAction>>>,
                                                             InstantiatedSublayer<Tag2, ElementMulLayer<PolicyContainer<PTanhAction>>>,
                                                             InstantiatedSublayer<Tag3, BiasLayer<PolicyContainer<PBatchMode, PTanhAction>>>,
                                                             InstantiatedSublayer<Tag4, TanhLayer<PolicyContainer<PTanhAction>>>,
                                                             InstantiatedSublayer<Tag5, AddLayer<PolicyContainer<PTanhAction>>>>;
    static_assert(std::is_same<Instantiation2, InstantiationComp2>::value, "Check Error");

    using Policy3 = PolicyContainer<PTanhAction, SubPolicyContainer<Tag2, PUpdate>>;
    using Instantiation3 = SublayerInstantiation<Policy3, check1, InterConnects>::type;
    using InstantiationComp3 = std::tuple<InstantiatedSublayer<Tag6, AddLayer<PolicyContainer<PTanhAction>>>,
                                                             InstantiatedSublayer<Tag1, AddLayer<PolicyContainer<PTanhAction>>>,
                                                             InstantiatedSublayer<Tag2, ElementMulLayer<PolicyContainer<PUpdate, PTanhAction>>>,
                                                             InstantiatedSublayer<Tag3, BiasLayer<PolicyContainer<PTanhAction, PFeedbackOutput>>>,
                                                             InstantiatedSublayer<Tag4, TanhLayer<PolicyContainer<PTanhAction, PFeedbackOutput>>>,
                                                             InstantiatedSublayer<Tag5, AddLayer<PolicyContainer<PTanhAction, PFeedbackOutput>>>>;
    static_assert(std::is_same<Instantiation3, InstantiationComp3>::value, "Check Error");

    using Policy4 = PolicyContainer<PTanhAction, SubPolicyContainer<Tag3, PUpdate>>;
    using Instantiation4 = SublayerInstantiation<Policy4, check1, InterConnects>::type;
    using InstantiationComp4 = std::tuple<InstantiatedSublayer<Tag6, AddLayer<PolicyContainer<PTanhAction>>>,
                                                             InstantiatedSublayer<Tag1, AddLayer<PolicyContainer<PTanhAction>>>,
                                                             InstantiatedSublayer<Tag2, ElementMulLayer<PolicyContainer<PTanhAction>>>,
                                                             InstantiatedSublayer<Tag3, BiasLayer<PolicyContainer<PUpdate, PTanhAction>>>,
                                                             InstantiatedSublayer<Tag4, TanhLayer<PolicyContainer<PTanhAction>>>,
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
    test_compose_kernel13();
    test_compose_kernel14();
}
