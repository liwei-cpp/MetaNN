#pragma once
#include <MetaNN/facilities/cont_metafuns/_.h>
#include <MetaNN/layers/facilities/make_layer.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/layer_io_map.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/change_policy.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/policies/policy_selector.h>
#include <MetaNN/layers/elementary/add_layer.h>
#include <MetaNN/layers/elementary/multiply_layer.h>
#include <type_traits>

namespace MetaNN
{
template <typename TLayerName, template<typename, typename> class TLayer>
struct Sublayer
{
    using LayerName = TLayerName;

    template <typename TInputs, typename TPolicies>
    using LayerType = TLayer<TInputs, TPolicies>;
};

template <typename TOutLayerName, typename TOutPort, typename TInLayerName, typename TInPort>
struct InternalConnect
{
    using OutLayer = TOutLayerName;
    using OutPort = TOutPort;
    using InLayer = TInLayerName;
    using InPort = TInPort;
};

template <typename TInPort, typename TInLayerName, typename TInLayerPort>
struct InConnect
{
    using InPort = TInPort;
    using InLayerName = TInLayerName;
    using InLayerPort = TInLayerPort;
};

template <typename TOutLayerName, typename TOutLayerPort, typename TOutPort>
struct OutConnect
{
    using OutLayerName = TOutLayerName;
    using OutLayerPort = TOutLayerPort;
    using OutPort = TOutPort;
};

namespace NSComposeKernel
{
template <typename TLayerName, typename TLayerType>
struct InstantiatedSublayer
{
    using LayerName = TLayerName;
    using LayerType = TLayerType;
};

template <typename...T> struct ClauseSeq;

/// ======================== Separate clauses ========================
template <typename...TClauses>
struct SeparateClauses_
{
    template<typename TS, typename TI, typename TO, 
             typename TIt, typename... T>
    struct imp
    {
        static_assert(sizeof...(T) == 0);

        using SublayerRes = TS;
        using InConnectRes = TI;
        using OutConnectRes = TO;
        using InterConnectRes = TIt;
    };
    
    template <typename TS, typename TI, typename TO, typename TIt, 
              typename TLayerName, template<typename, typename> class TLayer, typename...T>
    struct imp<TS, TI, TO, TIt, Sublayer<TLayerName, TLayer>, T...>
            : imp<ContMetaFun::Sequential::PushBack<TS, Sublayer<TLayerName, TLayer>>,
                  TI, TO, TIt,
                  T...>
    {};
    
    template <typename TS, typename TI, typename TO, typename TIt,
              typename TInPort, typename TInLayerName, typename TInLayerPort,
              typename...T>
    struct imp<TS, TI, TO, TIt, InConnect<TInPort, TInLayerName, TInLayerPort>, T...>
            : imp<TS,
                  ContMetaFun::Sequential::PushBack<TI, InConnect<TInPort, TInLayerName, TInLayerPort>>,
                  TO, TIt,
                  T...>
    {};
    
    template <typename TS, typename TI, typename TO, typename TIt,
              typename TOutLayerName, typename TOutLayerPort, typename TOutPort,
              typename...T>
    struct imp<TS, TI, TO, TIt, OutConnect<TOutLayerName, TOutLayerPort, TOutPort>, T...>
            : imp<TS, TI,
                  ContMetaFun::Sequential::PushBack<TO, OutConnect<TOutLayerName, TOutLayerPort, TOutPort>>,
                  TIt,
                  T...>
    {};
    
    template <typename TS, typename TI, typename TO, typename TIt, 
              typename TOutLayerName, typename TOutPort, typename TInLayerName, typename TInPort,
              typename...T>
    struct imp<TS, TI, TO, TIt, 
               InternalConnect<TOutLayerName, TOutPort, TInLayerName, TInPort>, T...>
            : imp<TS, TI, TO,
                  ContMetaFun::Sequential::PushBack<TIt, InternalConnect<TOutLayerName, TOutPort, TInLayerName, TInPort>>,
                  T...>
    { };
    
    using tmp = imp<ClauseSeq<>, ClauseSeq<>, ClauseSeq<>, ClauseSeq<>, TClauses...>;

    using SublayerRes = typename tmp::SublayerRes;
    using InterConnectRes = typename tmp::InterConnectRes;
    using InConnectRes = typename tmp::InConnectRes;
    using OutConnectRes = typename tmp::OutConnectRes;
};

namespace ClauseRefine
{
/// InternalFMap
template <typename TState, typename TInput>
using InternalFMap_ = ContMetaFun::MultiMap::Insert_<TState, typename TInput::OutLayer, TInput>;

template <typename TInternalClauses>
using InternalFMap = ContMetaFun::Sequential::Fold<ClauseSeq<>, TInternalClauses,
                                                   InternalFMap_>;

/// InternalBMap
template <typename TState, typename TInput>
using InternalBMap_ = ContMetaFun::MultiMap::Insert_<TState, typename TInput::InLayer, TInput>;

template <typename TInternalClauses>
using InternalBMap = ContMetaFun::Sequential::Fold<ClauseSeq<>, TInternalClauses,
                                                   InternalBMap_>;

/// InputFMap
template <typename TState, typename TInput>
using InputFMap_ = ContMetaFun::MultiMap::Insert_<TState, typename TInput::InLayerName, TInput>;

template <typename TInputClauses>
using InputFMap = ContMetaFun::Sequential::Fold<ClauseSeq<>, TInputClauses,
                                                InputFMap_>;

/// OutputBMap
template <typename TState, typename TOutput>
using OutputBMap_ = ContMetaFun::MultiMap::Insert_<TState, typename TOutput::OutLayerName, TOutput>;

template <typename TOutputClauses>
using OutputBMap = ContMetaFun::Sequential::Fold<ClauseSeq<>, TOutputClauses,
                                                 OutputBMap_>;

/// SublayerNameSet
template <typename TState, typename TInput>
using SublayerNameSet_ = ContMetaFun::Set::Insert_<TState, typename TInput::LayerName, true>;
    
template <typename TSublayerClauses>
using SublayerNameSet = ContMetaFun::Sequential::Fold<ClauseSeq<>, TSublayerClauses,
                                                      SublayerNameSet_>;

template <typename TSublayer>
struct SublayerNamePicker_
{
    using type = typename TSublayer::LayerName;
};

template <typename TSublayerClauses>
using SublayerMap = ContMetaFun::Map::CreateFromItems<TSublayerClauses, SublayerNamePicker_, ClauseSeq>;

/// InternalLayerSet
template <typename TState, typename TInput>
using InternalInLayerSet_ = ContMetaFun::Set::Insert_<TState, typename TInput::InLayer, true>;

template <typename TInternalClauses>
using InternalInLayerSet = ContMetaFun::Sequential::Fold<ClauseSeq<>, TInternalClauses,
                                                         InternalInLayerSet_>;

template <typename TState, typename TInput>
using InternalOutLayerSet_ = ContMetaFun::Set::Insert_<TState, typename TInput::OutLayer, true>;

template <typename TInternalClauses>
using InternalOutLayerSet = ContMetaFun::Sequential::Fold<ClauseSeq<>, TInternalClauses,
                                                          InternalOutLayerSet_>;

template <typename TInternalClauses>
using InternalLayerSet = ContMetaFun::Sequential::Fold<InternalInLayerSet<TInternalClauses>, TInternalClauses,
                                                       InternalOutLayerSet_>;
                                                       
/// InputNamePortSet
template <typename TState, typename TInput>
using InternalInNamePortSet_ = ContMetaFun::Set::Insert_<TState,
                                                         ContMetaFun::Helper::Pair<typename TInput::InLayer,
                                                                                   typename TInput::InPort>,
                                                         true>;

template <typename TState, typename TInput>
using InputNamePortSet_ = ContMetaFun::Set::Insert_<TState, 
                                                    ContMetaFun::Helper::Pair<typename TInput::InLayerName,
                                                                              typename TInput::InLayerPort>,
                                                    true>;

template <typename TInternalClauses>
using InternalInNamePortSet = ContMetaFun::Sequential::Fold<ClauseSeq<>, TInternalClauses,
                                                            InternalInNamePortSet_>;

template <typename TInternalClauses, typename TInputClauses>
using InputNamePortSet = ContMetaFun::Sequential::Fold<InternalInNamePortSet<TInternalClauses>, TInputClauses,
                                                       InputNamePortSet_>;

/// InputLayerSet
template <typename TState, typename TInput>
using InputLayerSet_ = ContMetaFun::Set::Insert_<TState, typename TInput::InLayerName, true>;

template <typename TInputClauses>
using InputLayerSet = ContMetaFun::Sequential::Fold<ClauseSeq<>, TInputClauses,
                                                    InputLayerSet_>;
                                                       
/// OutputPortSet
template <typename TState, typename TInput>
using OutputPortSet_ = ContMetaFun::Set::Insert_<TState, typename TInput::OutPort, true>;

template <typename TOutputClauses>
using OutputPortSet = ContMetaFun::Sequential::Fold<ClauseSeq<>, TOutputClauses,
                                                    OutputPortSet_>;
                                                    
/// OutputLayerSet
template <typename TState, typename TInput>
using OutputLayerSet_ = ContMetaFun::Set::Insert_<TState, typename TInput::OutLayerName, true>;

template <typename TInternalClauses>
using OutputLayerSet = ContMetaFun::Sequential::Fold<ClauseSeq<>, TInternalClauses,
                                                     OutputLayerSet_>;
}

/// ========= Internal Tag Should In Sublayer ==========================================
template <typename TInternalClauses, typename TSublayerNameSet>
constexpr bool InternalTagInSublayer = false;

template <typename...TInterTags, typename TSublayerNameSet>
constexpr bool InternalTagInSublayer<ClauseSeq<TInterTags...>, TSublayerNameSet> =
    (MetaNN::ContMetaFun::Set::HasKey<TSublayerNameSet, typename TInterTags::InLayer> && ...) &&
    (MetaNN::ContMetaFun::Set::HasKey<TSublayerNameSet, typename TInterTags::OutLayer> && ...);

/// ========= Input Tag Should In Sublayer =============================================
template <typename TInputClauses, typename TSublayerNameSet>
constexpr bool InputTagInSubLayer = false;

template <typename...TInputTags, typename TSublayerNameSet>
constexpr bool InputTagInSubLayer<ClauseSeq<TInputTags...>, TSublayerNameSet> = 
    (MetaNN::ContMetaFun::Set::HasKey<TSublayerNameSet, typename TInputTags::InLayerName> && ...);

/// ========= Output Tag Should In Sublayer ============================================
template <typename TOutputClauses, typename TSublayerNameSet>
constexpr bool OutputTagInSubLayer = false;

template <typename...TOutputTags, typename TSublayerNameSet>
constexpr bool OutputTagInSubLayer<ClauseSeq<TOutputTags...>, TSublayerNameSet> = 
    (MetaNN::ContMetaFun::Set::HasKey<TSublayerNameSet, typename TOutputTags::OutLayerName> && ...);
    
/// ========= Sublayers Tags Sould Exist in Other Sets =================================
template <typename TSublayerNameSet,
          typename TInterLayerSet, typename TInLayerSet, typename TOutLayerSet>
constexpr bool SublayerTagInOtherArrays = false;

template <typename...TSublayerElems,
          typename TInterLayerSet, typename TInLayerSet, typename TOutLayerSet>
constexpr bool SublayerTagInOtherArrays<ClauseSeq<TSublayerElems...>,
                                        TInterLayerSet, TInLayerSet, TOutLayerSet> = 
    ((MetaNN::ContMetaFun::Set::HasKey<TInterLayerSet, TSublayerElems> ||
      MetaNN::ContMetaFun::Set::HasKey<TInLayerSet, TSublayerElems> ||
      MetaNN::ContMetaFun::Set::HasKey<TOutLayerSet, TSublayerElems>) && ...);

/// ========= Topological Ordering Implementation ======================================
namespace NSTPO
{
    template <typename TInterLayers, typename TSublayerOrdered, typename TSubLayerUnordered, typename...T>
    struct SublayerPreprocess_
    {
        using Ordered = TSublayerOrdered;
        using Unordered = TSubLayerUnordered;
    };
    
    template <typename TInterLayers, typename TSublayerOrdered, typename TSubLayerUnordered, typename TCur, typename...T>
    struct SublayerPreprocess_<TInterLayers, 
                               TSublayerOrdered,
                               TSubLayerUnordered,
                               TCur, T...>
    {
        static constexpr bool inInter = MetaNN::ContMetaFun::Set::HasKey<TInterLayers, typename TCur::LayerName>;
        
        using NewOrdered = typename std::conditional_t<inInter,
                                                       Identity_<TSublayerOrdered>,
                                                       ContMetaFun::Sequential::PushBack_<TSublayerOrdered, typename TCur::LayerName>>::type;
        using NewUnordered = typename std::conditional_t<inInter,
                                                         ContMetaFun::Sequential::PushBack_<TSubLayerUnordered, typename TCur::LayerName>,
                                                         Identity_<TSubLayerUnordered>>::type;
                                                         
        using Ordered = typename SublayerPreprocess_<TInterLayers, NewOrdered, NewUnordered, T...>::Ordered;
        using Unordered = typename SublayerPreprocess_<TInterLayers, NewOrdered, NewUnordered, T...>::Unordered;
    };
    
    template <typename TCheckInterSet, typename TRemainInters, typename TPostTags, typename...T>
    struct InternalLayerPrune
    {
        using PostTags = TPostTags;
        using RemainIters = TRemainInters;
    };
    
    template <typename TCheckInterSet, typename TRemainInters, typename TPostTags, typename TCur, typename...T3>
    struct InternalLayerPrune<TCheckInterSet,
                              TRemainInters,
                              TPostTags,
                              TCur, T3...>
    {
        static constexpr bool inInterIn = ContMetaFun::Set::HasKey<TCheckInterSet, typename TCur::OutLayer>;
        
        using NewRemainInters = typename std::conditional_t<inInterIn,
                                                            ContMetaFun::Sequential::PushBack_<TRemainInters, TCur>,
                                                            Identity_<TRemainInters>>::type;
        using NewTagContainer = typename std::conditional_t<inInterIn,
                                                            Identity_<TPostTags>,
                                                            ContMetaFun::Set::Insert_<TPostTags, typename TCur::OutLayer, true>>::type;

        using nextStep = InternalLayerPrune<TCheckInterSet, NewRemainInters, NewTagContainer, T3...>;

        using RemainIters = typename nextStep::RemainIters;
        using PostTags = typename nextStep::PostTags;
    };
    
    template <typename TOrderedSublayers, typename TUnorderedSublayers,
              typename TCheckInternals>
    struct MainLoop
    {
        static_assert((ArraySize<TCheckInternals> == 0), "Cycle exist in the compose layer");
        using type = ContMetaFun::Sequential::Cascade<TOrderedSublayers, TUnorderedSublayers>;
    };
    
    template <typename TOrderedSublayers, typename TUnorderedSublayers, typename TIC, typename...TI>
    struct MainLoop<TOrderedSublayers,
                    TUnorderedSublayers,
                    ClauseSeq<TIC, TI...>>
    {
        using InternalLayerPruneRes = 
            InternalLayerPrune<ClauseRefine::InternalInLayerSet<ClauseSeq<TIC, TI...>>,
                               ClauseSeq<>,
                               ClauseSeq<>, TIC, TI...>;

        using NewInter = typename InternalLayerPruneRes::RemainIters;
        using PostTags = typename InternalLayerPruneRes::PostTags;
        static_assert((ArraySize<NewInter> < ArraySize<ClauseSeq<TIC, TI...>>),
                      "Cycle exist in the compose layer");

        using NewOrdered = ContMetaFun::Sequential::Cascade<TOrderedSublayers, PostTags>;
        using NewUnordered = ContMetaFun::Sequential::Fold<TUnorderedSublayers, PostTags,
                                                           ContMetaFun::Set::Erase_>;
        using type = typename MainLoop<NewOrdered, NewUnordered, NewInter>::type;
    };
}

template <typename TSublayerClause, typename TInterClause>
struct TopologicalOrdering_;

template <typename...TSublayers, typename TInterClause>
struct TopologicalOrdering_<ClauseSeq<TSublayers...>, TInterClause>
{
    using SublayerPreRes = NSTPO::SublayerPreprocess_<ClauseRefine::InternalLayerSet<TInterClause>,
                                                      ClauseSeq<>, ClauseSeq<>, TSublayers...>;

    using type = typename NSTPO::MainLoop<typename SublayerPreRes::Ordered,
                                          typename SublayerPreRes::Unordered,
                                          TInterClause>::type;
};

namespace NSSI
{
    // Assign policy
    template <typename TPolicy>
    struct GetSublayerPolicy_
    {
        template <typename TSublayerName>
        struct apply
        {
            using type = ContMetaFun::Helper::KVBinder<TSublayerName, SubPolicyPicker<TPolicy, TSublayerName>>;
        };
    };
    
    template <typename TInputPolicyCont, typename SublayerNameSeq, typename TInconnects>
    struct FbSetByInConnection_
    {
        using type = TInputPolicyCont;
    };
    
    template <typename TInputPolicyCont, typename SublayerNameSeq,
              template<typename...> class TInCont, typename TCur, typename... TItems>
    struct FbSetByInConnection_<TInputPolicyCont, SublayerNameSeq, TInCont<TCur, TItems...>>
    {
        constexpr static size_t pos = ContMetaFun::Sequential::Order<SublayerNameSeq, TCur>;
        using OriType = ContMetaFun::Sequential::At<TInputPolicyCont, pos>;
        using NewPolicy = ChangePolicy<PFeedbackOutput, typename OriType::ValueType>;
        using NewInputPolicyCont
            = ContMetaFun::Sequential::Set<TInputPolicyCont, pos,
                                           ContMetaFun::Helper::KVBinder<typename OriType::KeyType, NewPolicy>>;
        using type = typename FbSetByInConnection_<NewInputPolicyCont, SublayerNameSeq, TInCont<TItems...>>::type;
    };
    
    template <typename TInternalConnect>
    struct PickInLayerFromInternalConnect_
    {
        using type = typename TInternalConnect::InLayer;
    };
    
    template <typename TUpdateLayerSet>
    struct UpdatePolicyThroughInterMapHelper_
    {
        template <typename TSublayerPolicy>
        struct apply
        {
            using LayerName = typename TSublayerPolicy::KeyType;
            using OriPolicy = typename TSublayerPolicy::ValueType;
            constexpr static bool ShouldUpdate = ContMetaFun::Set::HasKey<TUpdateLayerSet, LayerName>;
            using NewPolicy = typename std::conditional_t<ShouldUpdate,
                                                          ChangePolicy_<PFeedbackOutput, OriPolicy>,
                                                          Identity_<OriPolicy>>::type;
            using type = ContMetaFun::Helper::KVBinder<LayerName, NewPolicy>;
        };
    };

    template <typename TOutLayerName, typename TSublayerPolicies, typename TInterMap>
    struct UpdatePolicyThroughInterMap_
    {
        using LayerProcs = ContMetaFun::MultiMap::Find<TInterMap, TOutLayerName>;
        using UpdateLayerSet = ContMetaFun::Set::CreateFromItems<LayerProcs, PickInLayerFromInternalConnect_, true>;
        using NewSublayerPolicies
            = ContMetaFun::Sequential::Transform<TSublayerPolicies,
                                                 UpdatePolicyThroughInterMapHelper_<UpdateLayerSet>::template apply,
                                                 std::tuple>;
        using type = NewSublayerPolicies;
    };

    template <typename TProcessed, typename TRemain, typename InterFMap>
    struct FbSetByInternalConnection_
    {
        using type = TProcessed;
    };

    template <typename TProcessed, typename TCur, typename...TInstElements, typename InterFMap>
    struct FbSetByInternalConnection_<TProcessed, std::tuple<TCur, TInstElements...>, InterFMap>
    {
        constexpr static bool isPolUpdate = PolicySelect<GradPolicy, typename TCur::ValueType>::IsFeedbackOutput ||
                                            PolicySelect<GradPolicy, typename TCur::ValueType>::IsUpdate;

        using NewProcessed = ContMetaFun::Sequential::PushBack<TProcessed, TCur>;

        using NewRemain = typename std::conditional_t<isPolUpdate && (sizeof...(TInstElements) != 0),
                                                      UpdatePolicyThroughInterMap_<typename TCur::KeyType,
                                                                                   std::tuple<TInstElements...>,
                                                                                   InterFMap>,
                                                      Identity_<std::tuple<TInstElements...>>>::type;
        using type = typename FbSetByInternalConnection_<NewProcessed, NewRemain, InterFMap>::type;
    };
    
    template <typename TOutputCont, typename TDataMap, typename TSeq>
    struct InputTypeFillInConnectHelper_
    {
        using type = TOutputCont;
    };
    
    template <typename TOutputCont, typename TDataMap, typename TCur, typename... TRemain>
    struct InputTypeFillInConnectHelper_<TOutputCont, TDataMap, ContMetaFun::Helper::ValueSequence<TCur, TRemain...>>
    {
        using FromPort = typename TCur::InPort;
        using ToPort = typename TCur::InLayerPort;
        using AimType = typename TDataMap::template Find<FromPort>;
        using NewCont = ContMetaFun::Sequential::PushBack<TOutputCont, LayerKV<ToPort, AimType>>;
        using type = typename InputTypeFillInConnectHelper_<NewCont, TDataMap, ContMetaFun::Helper::ValueSequence<TRemain...>>::type;
    };
    
    template <typename TSublayerCont, typename TInFMap, typename TOutputCont, typename TInputDataMap>
    struct InputTypeFillInConnect_
    {
        using type = TOutputCont;
    };
    
    template <typename TInFMap, typename TOutputCont, typename TInputDataMap, typename TCur, typename... TRemain>
    struct InputTypeFillInConnect_<ClauseSeq<TCur, TRemain...>, TInFMap, TOutputCont, TInputDataMap>
    {
        using CurValueSeq = ContMetaFun::MultiMap::Find<TInFMap, TCur>;
        using TCurDataMap = typename InputTypeFillInConnectHelper_<LayerIOMap<>, TInputDataMap, CurValueSeq>::type;
        using NewOutputCont = ContMetaFun::Sequential::PushBack<TOutputCont, TCurDataMap>;
        using type = typename InputTypeFillInConnect_<ClauseSeq<TRemain...>, TInFMap, NewOutputCont, TInputDataMap>::type;
    };

    template <typename TSublayerOutputMap, typename TConnectInfo, typename SublayerSeq, typename TOutputCont>
    struct InputTypeFillInternalConnectHelper2_
    {
        using type = TOutputCont;
    };
    
    template <typename TCur, typename... TRemain, typename TSublayerOutputMap, typename SublayerSeq, typename TOutputCont>
    struct InputTypeFillInternalConnectHelper2_<TSublayerOutputMap, ContMetaFun::Helper::ValueSequence<TCur, TRemain...>,
                                                SublayerSeq, TOutputCont>
    {
        using InLayer = typename TCur::InLayer;
        constexpr static size_t pos = ContMetaFun::Sequential::Order<SublayerSeq, InLayer>;
        using OriOutput = ContMetaFun::Sequential::At<TOutputCont, pos>;
        
        using OutPort = typename TCur::OutPort;
        using AimType = typename TSublayerOutputMap::template Find<OutPort>;
        
        using InPort = typename TCur::InPort;
        static_assert(std::is_same_v<typename OriOutput::template Find<InPort>, NullParameter>);
        
        using NewOutput = ContMetaFun::Sequential::PushBack<OriOutput, LayerKV<InPort, AimType>>;
        using NewOutputCont = ContMetaFun::Sequential::Set<TOutputCont, pos, NewOutput>;
        
        using type
            = typename InputTypeFillInternalConnectHelper2_<TSublayerOutputMap, ContMetaFun::Helper::ValueSequence<TRemain...>,
                                                            SublayerSeq, NewOutputCont>::type;
    };
    
    template <typename TLayerTemp, typename TInputMap, typename TPolicy, typename SublayerSeq, typename TConnectInfo, typename TOutput>
    struct InputTypeFillInternalConnectHelper_
    {
        using LayerType = MakeTrainLayer<TLayerTemp::template LayerType, TInputMap, TPolicy>;
        using type = typename InputTypeFillInternalConnectHelper2_<LayerTraits::LayerOutputItemTypes<LayerType>,
                                                                   TConnectInfo, SublayerSeq, TOutput>::type;
    };
    
    template <size_t CurID, size_t MAXID,
              typename TSublayerNameCont, typename TInputMapCont, typename TPolicyCont,
              typename TSublayerMap, typename TInternalMap>
    struct InputTypeFillInternalConnect_
    {
        static_assert(CurID < MAXID);
        using TCurLayer = ContMetaFun::Sequential::At<TSublayerNameCont, CurID>;
        using InternalConnections = ContMetaFun::MultiMap::Find<TInternalMap, TCurLayer>;
        
        using NewInputMapCont
            = typename std::conditional_t<(ArraySize<InternalConnections> == 0),
                                          Identity_<TInputMapCont>,
                                          InputTypeFillInternalConnectHelper_<ContMetaFun::Map::Find<TSublayerMap, TCurLayer>,
                                                                              ContMetaFun::Sequential::At<TInputMapCont, CurID>,
                                                                              ContMetaFun::Map::Find<TPolicyCont, TCurLayer>,
                                                                              TSublayerNameCont,
                                                                              InternalConnections, TInputMapCont>>::type;
        using type = typename InputTypeFillInternalConnect_<CurID + 1, MAXID,
                                                            TSublayerNameCont, NewInputMapCont, TPolicyCont,
                                                            TSublayerMap, TInternalMap>::type;
    };
    
    template <size_t MAXID,
              typename TSublayerNameCont, typename TInputMapCont, typename TPolicyCont,
              typename TSublayerMap, typename TInternalMap>
    struct InputTypeFillInternalConnect_<MAXID, MAXID, TSublayerNameCont, TInputMapCont, TPolicyCont,
                                         TSublayerMap, TInternalMap>
    {
        using type = TInputMapCont;
    };
    
    template <typename TOutCont, typename TLayerNames, typename TLayerMap, typename TInputMap, typename TPolicies>
    struct Instantiation_
    {
        using type = TOutCont;
    };
    
    template <typename TCurLayer, typename... TRemainLayers, typename TInputHead, typename... TRemainInputMap,
              typename TLayerMap, typename TPolicies, typename TOutCont>
    struct Instantiation_<TOutCont, ClauseSeq<TCurLayer, TRemainLayers...>, TLayerMap, std::tuple<TInputHead, TRemainInputMap...>,
                          TPolicies>
    {
        using TLayerTemp = ContMetaFun::Map::Find<TLayerMap, TCurLayer>;
        static_assert(!std::is_same_v<TLayerTemp, void>);
        using TCurPolicy = ContMetaFun::Map::Find<TPolicies, TCurLayer>;
        static_assert(!std::is_same_v<TCurPolicy, void>);
        using LayerType = typename TLayerTemp::template LayerType<TInputHead, TCurPolicy>;
        using NewOutCont = ContMetaFun::Sequential::PushBack<TOutCont, LayerType>;
        using type = typename Instantiation_<NewOutCont, ClauseSeq<TRemainLayers...>, TLayerMap, std::tuple<TRemainInputMap...>,
                                             TPolicies>::type;
    };
    
    template <typename TInputs, typename OrderedSublayers, typename TSublayerClauses, typename InConnects, typename InterConnects, typename SublayerPolicyFinal>
    struct NontrivalInst_
    {
        //  Fill Input type container with in-connections
        using InputTypeCont1 = typename NSSI::InputTypeFillInConnect_<OrderedSublayers,
                                                                      ClauseRefine::InputFMap<InConnects>,
                                                                      std::tuple<>, TInputs>::type;
        // Fill Input type container with inter-connections
        using InputTypeContFinal = typename NSSI::InputTypeFillInternalConnect_<0, ArraySize<OrderedSublayers>, OrderedSublayers,
                                                                                InputTypeCont1, SublayerPolicyFinal,
                                                                                ClauseRefine::SublayerMap<TSublayerClauses>,
                                                                                ClauseRefine::InternalFMap<InterConnects>>::type;
        /// Instantiation
        using type = typename NSSI::Instantiation_<std::tuple<>, OrderedSublayers, ClauseRefine::SublayerMap<TSublayerClauses>,
                                                   InputTypeContFinal, SublayerPolicyFinal>::type;
    };
    
    template <typename TSublayerMap, typename SublayerPolicyFinal>
    struct TrivalInstHelper_
    {
        template <typename TCurLayer>
        struct apply
        {
            using SublayerInfo = ContMetaFun::Map::Find<TSublayerMap, TCurLayer>;
            using SublayerPolicy = ContMetaFun::Map::Find<SublayerPolicyFinal, TCurLayer>;
            using type = MakeInferLayer<SublayerInfo::template LayerType, SublayerPolicy>;
        };
    };
    
    template <typename OrderedSublayers, typename TSublayerClauses, typename SublayerPolicyFinal>
    struct TrivalInst_
    {
        using type = ContMetaFun::Sequential::Transform<OrderedSublayers,
                                                        TrivalInstHelper_<ClauseRefine::SublayerMap<TSublayerClauses>,
                                                                          SublayerPolicyFinal>::template apply,
                                                        std::tuple>;
    };
}

template <typename TInputs, typename TPolicies,
          typename OrderedSublayers, typename TSublayerClauses, typename InConnects, typename InterConnects, typename OutConnects>
struct SublayerInstantiation_
{
    static_assert(IsPolicyContainer<TPolicies>, "Not a Policy Container");
    
    using SublayerPolicy1 = ContMetaFun::Sequential::Transform<OrderedSublayers,
                                                               NSSI::GetSublayerPolicy_<TPolicies>::template apply,
                                                               std::tuple>;

    /// Policy modification
    //  if feedbackout is set in parent layer, then each sublayer that includes in InConnects should also set it to true
    constexpr static bool IsPlainPolicyFeedbackOut = PolicySelect<GradPolicy, PlainPolicy<TPolicies>>::IsFeedbackOutput;
    using SublayerPolicy2
        =typename std::conditional_t<!IsPlainPolicyFeedbackOut,
                                      Identity_<SublayerPolicy1>,
                                      NSSI::FbSetByInConnection_<SublayerPolicy1, OrderedSublayers,
                                                                 ClauseRefine::InputLayerSet<InConnects>>>::type;
                                         
    /// for any instance A, if there is a connection A->B and A is feedbackin, then B should set to feedbackout
    using SublayerPolicyFinal = typename NSSI::FbSetByInternalConnection_<std::tuple<>,
                                                                          SublayerPolicy2,
                                                                          ClauseRefine::InternalFMap<InterConnects>>::type;

    /// Instantiation
    using type = typename std::conditional_t<IsEmptyLayerIOMap<TInputs>,
                                             NSSI::TrivalInst_<OrderedSublayers, TSublayerClauses, SublayerPolicyFinal>,
                                             NSSI::NontrivalInst_<TInputs, OrderedSublayers, TSublayerClauses, InConnects, InterConnects, SublayerPolicyFinal>>::type;

};
}

template <typename...TComposeClauses>
struct ComposeTopology
{
/// ========== Separate Results ========================================
    using Sublayers = typename NSComposeKernel::SeparateClauses_<TComposeClauses...>::SublayerRes;
    using InterConnects = typename NSComposeKernel::SeparateClauses_<TComposeClauses...>::InterConnectRes;
    using InputConnects = typename NSComposeKernel::SeparateClauses_<TComposeClauses...>::InConnectRes;
    using OutputConnects = typename NSComposeKernel::SeparateClauses_<TComposeClauses...>::OutConnectRes;
    
    using SublayerNameSet = NSComposeKernel::ClauseRefine::SublayerNameSet<Sublayers>;
    using InternalFMap = NSComposeKernel::ClauseRefine::InternalFMap<InterConnects>;
    
/// ========== Asserts =================================================
    static_assert((ArraySize<Sublayers> != 0), "Sublayer is empty.");
    static_assert((ArraySize<Sublayers> == ArraySize<SublayerNameSet>), "Two or more sublayers have same tag.");
    static_assert(ArraySize<InputConnects> + ArraySize<InterConnects> ==
                  ArraySize<NSComposeKernel::ClauseRefine::InputNamePortSet<InterConnects, InputConnects>>,
                  "One input corresponds to two or more sources.");
    static_assert(ArraySize<OutputConnects> == ArraySize<NSComposeKernel::ClauseRefine::OutputPortSet<OutputConnects>>,
                  "One output corresponds to two or more sources.");
    static_assert(NSComposeKernel::InternalTagInSublayer<InterConnects, SublayerNameSet>,
                  "Internal connections have tags are not sublayer tags.");
    static_assert(NSComposeKernel::InputTagInSubLayer<InputConnects, SublayerNameSet>,
                  "One or more input tags are not sublayer tags.");
    static_assert(NSComposeKernel::OutputTagInSubLayer<OutputConnects, SublayerNameSet>,
                  "One or more output tags are not sublayer tags.");
    static_assert(NSComposeKernel::SublayerTagInOtherArrays<SublayerNameSet,
                                                            NSComposeKernel::ClauseRefine::InternalLayerSet<InterConnects>,
                                                            NSComposeKernel::ClauseRefine::InputLayerSet<InputConnects>,
                                                            NSComposeKernel::ClauseRefine::OutputLayerSet<OutputConnects>>,
                  "One ore more sublayer tags not belong to any connection containers.");
/// ========== Topological Ordering ===================================
    using TopologicalOrdering = typename NSComposeKernel::TopologicalOrdering_<Sublayers, InterConnects>::type;
    
    template <typename TInputMap, typename TPolicyCont>
    using Instances
        = typename NSComposeKernel::SublayerInstantiation_<TInputMap, TPolicyCont,
                                                           TopologicalOrdering, Sublayers,
                                                           InputConnects, InterConnects, OutputConnects>::type;
};

namespace NSComposeKernel
{
    template <typename TType>
    struct AddSharedPtrWrapper_
    {
        using type = std::shared_ptr<TType>;
    };
    
    template <typename TLayerInsts>
    struct IsComposeLayerUpdate_;
    
    template <typename... TLayerInst>
    struct IsComposeLayerUpdate_<std::tuple<TLayerInst...>>
    {
        constexpr static bool value = (TLayerInst::IsUpdate || ...);
    };
    
    template <typename TSublayerNameTuple, typename TSublayerTuple>
    struct SublayerArrayMaker
    {
        static_assert(ArraySize<TSublayerNameTuple> == ArraySize<TSublayerTuple>);
    public:
        using SublayerArray
            = ContMetaFun::Sequential::Transform<TSublayerTuple, NSComposeKernel::AddSharedPtrWrapper_, std::tuple>;

    private:
        template <size_t N>
        void FillGap()
        {
            if constexpr (N == ArraySize<TSublayerTuple>)
            {
                return;
            }
            else
            {
                using AimType = typename ContMetaFun::Sequential::At<TSublayerTuple, N>;
                if constexpr (std::is_default_constructible_v<AimType>)
                {
                    if (!std::get<N>(m_tuple))
                    {
                        std::get<N>(m_tuple) = std::make_shared<AimType>();
                    }
                }
                
                if (!std::get<N>(m_tuple))
                {
                    throw std::runtime_error("Sublayer not initialized");
                }
                FillGap<N + 1>();
            }
        }
        
    public:
        template <typename TLayerName, typename...TParams>
        auto Set(TParams&&... params)
        {
            constexpr static size_t Pos = ContMetaFun::Sequential::Order<TSublayerNameTuple, TLayerName>;
            
            using AimType = typename ContMetaFun::Sequential::At<TSublayerTuple, Pos>;
            std::get<Pos>(m_tuple) = std::make_shared<AimType>(std::forward<TParams>(params)...);
            return *this;
        }
        
        operator SublayerArray()
        {
            FillGap<0>();
            return m_tuple;
        }
        
    private:
        SublayerArray m_tuple;
    };
    
    template <size_t N, typename TInitializer, typename TBuffer, typename TSublayers>
    void Init(TInitializer& initializer, TBuffer& loadBuffer, TSublayers& sublayers)
    {
        if constexpr (N != ArraySize<TSublayers>)
        {
            auto& layer = std::get<N>(sublayers);
            LayerInit(*layer, initializer, loadBuffer);
            Init<N + 1>(initializer, loadBuffer, sublayers);
        }
    }

    
    template <size_t N, typename TSave, typename TSublayers>
    void SaveWeights(TSave& saver, const TSublayers& sublayers)
    {
        if constexpr (N != ArraySize<TSublayers>)
        {
            auto& layer = std::get<N>(sublayers);
            LayerSaveWeights(*layer, saver);
            SaveWeights<N + 1>(saver, sublayers);
        }
    }
    
    template <size_t N, typename TGradCollector, typename TSublayers>
    void GradCollect(TGradCollector& collector, TSublayers& sublayers)
    {
        if constexpr (N != ArraySize<TSublayers>)
        {
            auto& layer = std::get<N>(sublayers);
            LayerGradCollect(*layer, collector);
            GradCollect<N + 1>(collector, sublayers);
        }
    }

    template <size_t N, typename TSublayers>
    void NeutralInvariant(TSublayers& sublayers)
    {
        if constexpr (N != ArraySize<TSublayers>)
        {
            auto& layer = std::get<N>(sublayers);
            LayerNeutralInvariant(*layer);
            NeutralInvariant<N + 1>(sublayers);
        }
    }
    
    template <typename TSublayers>
    struct InternalResult;
    
    template <typename... TSublayers>
    struct InternalResult<NSComposeKernel::ClauseSeq<TSublayers...>> : public VarTypeDict<TSublayers...> {};

    template <size_t N, typename TLayerNames, typename TLayerInst, typename TInput>
    auto CreateInputInternalBuf(TInput&& m_input)
    {
        if constexpr (N == ArraySize<TLayerInst>)
        {
            return std::forward<TInput>(m_input);
        }
        else
        {
            using TCurName = ContMetaFun::Sequential::At<TLayerNames, N>;
            using TCurInst = ContMetaFun::Sequential::At<TLayerInst, N>;
            auto inputCont = LayerInputCont<TCurInst>();
            auto newInput = std::move(m_input).template Set<TCurName>(std::move(inputCont));
            return CreateInputInternalBuf<N + 1, TLayerNames, TLayerInst>(std::move(newInput));
        }
    }
    
    template <size_t N, typename TLayerNames, typename TLayerInst, typename TOutput>
    auto CreateOutputInternalBuf(TOutput&& m_output)
    {
        if constexpr (N == ArraySize<TLayerInst>)
        {
            return std::forward<TOutput>(m_output);
        }
        else
        {
            using TCurName = ContMetaFun::Sequential::At<TLayerNames, N>;
            using TCurInst = ContMetaFun::Sequential::At<TLayerInst, N>;
            auto outputCont = LayerOutputCont<TCurInst>();
            auto newOutput = std::move(m_output).template Set<TCurName>(std::move(outputCont));
            return CreateOutputInternalBuf<N + 1, TLayerNames, TLayerInst>(std::move(newOutput));
        }
    }
    
    template <size_t N, typename TInputClauses, typename TIn, typename TInternal>
    auto FillInput(const TIn& p_in, TInternal&& p_internal)
    {
        if constexpr (N == ArraySize<TInputClauses>)
        {
            return std::move(p_internal);
        }
        else
        {
            using TCur = ContMetaFun::Sequential::At<TInputClauses, N>;
            auto source = p_in.template Get<typename TCur::InPort>();
            auto dest = p_internal.template Get<typename TCur::InLayerName>();
            
            auto fillRes = std::move(dest).template Set<typename TCur::InLayerPort>(std::move(source));
            
            auto newInternal = std::move(p_internal).template Set<typename TCur::InLayerName>(std::move(fillRes));
            return FillInput<N + 1, TInputClauses>(p_in, std::move(newInternal));
        }
    }
    
    template <size_t N, typename TMap, typename TForwardRes, typename TAim>
    auto ForwardFillInternal(const TForwardRes& input, TAim&& p_aim)
    {
        if constexpr (ArraySize<TMap> == 0)
        {
            return std::move(p_aim);
        }
        else 
        {
            if constexpr (N == ArraySize<TMap>)
            {
                return std::move(p_aim);
            }
            else
            {
                using TCur = ContMetaFun::Sequential::At<TMap, N>;
                
                auto value = input.template Get<typename TCur::OutPort>();
                
                auto des = std::move(p_aim).template Get<typename TCur::InLayer>();
                auto newDes = std::move(des).template Set<typename TCur::InPort>(std::move(value));
                
                auto newAim = std::move(p_aim).template Set<typename TCur::InLayer>(newDes);
                
                return ForwardFillInternal<N + 1, TMap>(input, std::move(newAim));
            }
        }
    }
    
    template <size_t N, typename TLayerInfo, typename TFMap,
              typename TSublayers, typename TInput, typename TOutput>
    auto FeedForward(TSublayers&& sublayers, TInput&& p_input, TOutput&& m_output)
    {
        if constexpr (N == ArraySize<TLayerInfo>)
        {
            return std::move(m_output);
        }
        else
        {
            using TCurLayerName = ContMetaFun::Sequential::At<TLayerInfo, N>;
            auto source = std::forward<TInput>(p_input).template Get<TCurLayerName>();
            auto forwardRes = std::get<N>(sublayers)->FeedForward(std::move(source));
            
            using ItemsFromMap = ContMetaFun::MultiMap::Find<TFMap, TCurLayerName>;
            
            auto newInput = ForwardFillInternal<0, ItemsFromMap>(forwardRes, std::move(p_input));
            auto newOutput = std::move(m_output).template Set<TCurLayerName>(forwardRes);
            
            return FeedForward<N+1, TLayerInfo, TFMap>(sublayers, std::move(newInput), std::move(newOutput));
        }
    }
    
    template <size_t N, typename TOutputClauses, typename TIn, typename TAim>
    auto FillOutput(const TIn& p_in, TAim&& p_aim)
    {
        if constexpr (N == ArraySize<TOutputClauses>)
        {
            return std::move(p_aim);
        }
        else
        {
            using TCur = ContMetaFun::Sequential::At<TOutputClauses, N>;
            
            auto sourceLayer = p_in.template Get<typename TCur::OutLayerName>();
            auto source = sourceLayer.template Get<typename TCur::OutLayerPort>();
            
            auto newAim = std::move(p_aim).template Set<typename TCur::OutPort>(std::move(source));
            return FillOutput<N + 1, TOutputClauses>(p_in, std::move(newAim));
        }
    }
    
    template <size_t N, typename TOutputClauses, typename TInGrad, typename TInternal>
    auto FillInputGrad(const TInGrad& p_inGrad, TInternal&& p_internal)
    {
        if constexpr (N == ArraySize<TOutputClauses>)
        {
            return std::move(p_internal);
        }
        else
        {
            using TCur = ContMetaFun::Sequential::At<TOutputClauses, N>;
            
            auto source = p_inGrad.template Get<typename TCur::OutPort>();
            auto dest = p_internal.template Get<typename TCur::OutLayerName>();
            auto prevInfo = dest.template Get<typename TCur::OutLayerPort>();
            if constexpr (std::is_same_v<RemConstRef<decltype(prevInfo)>, NullParameter>)
            {
                auto fillRes = std::move(dest).template Set<typename TCur::OutLayerPort>(std::move(source));
                auto newInternal = std::move(p_internal).template Set<typename TCur::OutLayerName>(std::move(fillRes));
                return FillInputGrad<N + 1, TOutputClauses>(p_inGrad, std::move(newInternal));
            }
            else
            {
                auto fillRes = std::move(dest).template Set<typename TCur::OutLayerPort>(prevInfo + source);
                auto newInternal = std::move(p_internal).template Set<typename TCur::OutLayerName>(std::move(fillRes));
                return FillInputGrad<N + 1, TOutputClauses>(p_inGrad, std::move(newInternal));
            }
        }
    }
    
    template <size_t N, typename TMap, typename TBackwardRes, typename TAim>
    auto BackwardFillInternal(const TBackwardRes& input, TAim&& p_aim)
    {
        if constexpr (ArraySize<TMap> == 0)
        {
            return std::move(p_aim);
        }
        else 
        {
            if constexpr (N == ArraySize<TMap>)
            {
                return std::move(p_aim);
            }
            else
            {
                using TCur = ContMetaFun::Sequential::At<TMap, N>;
                
                auto value = input.template Get<typename TCur::InPort>();
                
                auto des = std::move(p_aim).template Get<typename TCur::OutLayer>();
                auto prevInfo = std::move(des).template Get<typename TCur::OutPort>();
                if constexpr (std::is_same_v<RemConstRef<decltype(prevInfo)>, NullParameter>)
                {
                    auto newDes = std::move(des).template Set<typename TCur::OutPort>(std::move(value));
                    auto newAim = std::move(p_aim).template Set<typename TCur::OutLayer>(std::move(newDes));
                    return BackwardFillInternal<N + 1, TMap>(input, std::move(newAim));
                }
                else
                {
                    auto newDes = std::move(des).template Set<typename TCur::OutPort>(prevInfo + value);
                    auto newAim = std::move(p_aim).template Set<typename TCur::OutLayer>(std::move(newDes));
                    return BackwardFillInternal<N + 1, TMap>(input, std::move(newAim));
                }
                
            }
        }
    }
    
    template <size_t N, typename TLayerInfo, typename TBMap,
              typename TSublayers, typename TInputGrad, typename TOutput>
    auto FeedBackward(TSublayers&& sublayers, TInputGrad&& p_input, TOutput&& m_output)
    {
        if constexpr (N == 0)
        {
            return std::move(m_output);
        }
        else
        {
            using TCur = ContMetaFun::Sequential::At<TLayerInfo, N - 1>;
            auto source = std::move(p_input).template Get<typename TCur::LayerName>();
            auto backwardRes = std::get<N - 1>(sublayers)->FeedBackward(std::move(source));
            
            using ItemsFromMap = ContMetaFun::MultiMap::Find<TBMap, typename TCur::LayerName>;
            
            auto newInput = BackwardFillInternal<0, ItemsFromMap>(backwardRes, std::move(p_input));
            auto newOutput = std::move(m_output).template Set<typename TCur::LayerName>(backwardRes);
            
            return FeedBackward<N-1, TLayerInfo, TBMap>(sublayers, std::move(newInput), std::move(newOutput));
        }
    }
    
    template <size_t N, typename TInputClauses, typename TInGrad, typename TAim>
    auto FillOutputGrad(const TInGrad& p_in, TAim&& p_aim)
    {
        if constexpr (N == ArraySize<TInputClauses>)
        {
            return std::move(p_aim);
        }
        else
        {
            using TCur = ContMetaFun::Sequential::At<TInputClauses, N>;
            
            auto sourceLayer = p_in.template Get<typename TCur::InLayerName>();
            auto source = sourceLayer.template Get<typename TCur::InLayerPort>();
            
            auto prevInfo = std::move(p_aim).template Get<typename TCur::InPort>();
            
            if constexpr (std::is_same_v<RemConstRef<decltype(prevInfo)>, NullParameter>)
            {
                auto newAim = std::move(p_aim).template Set<typename TCur::InPort>(std::move(source));
                return FillOutputGrad<N + 1, TInputClauses>(p_in, std::move(newAim));
            }
            else
            {
                auto newAim = std::move(p_aim).template Set<typename TCur::InPort>(prevInfo + source);
                return FillOutputGrad<N + 1, TInputClauses>(p_in, std::move(newAim));
            }
        }
    }
};

template <typename TInputs, typename TPolicyCont, typename TKernelTopo>
class ComposeKernel
{
    static_assert(IsPolicyContainer<TPolicyCont>, "Parameter is not a policy container.");
    using PlainPolicies = PlainPolicy<TPolicyCont>;
    
private:
    using TOrderedSublayerSeq = typename TKernelTopo::TopologicalOrdering;
    using TSublayerInstCont = typename TKernelTopo::template Instances<TInputs, TPolicyCont>;
    using SublayerArray = typename NSComposeKernel::SublayerArrayMaker<TOrderedSublayerSeq, TSublayerInstCont>::SublayerArray;
    using InternalResult = NSComposeKernel::InternalResult<TOrderedSublayerSeq>;
    
public:
    static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, PlainPolicies>::IsFeedbackOutput;
    static constexpr bool IsUpdate = NSComposeKernel::IsComposeLayerUpdate_<TSublayerInstCont>::value;
    
    using InputMap = TInputs;

public:
    static auto CreateSublayers()
    {
        return NSComposeKernel::SublayerArrayMaker<TOrderedSublayerSeq, TSublayerInstCont>();
    }
    
public:
    ComposeKernel(SublayerArray p_sublayers)
        : sublayers(std::move(p_sublayers)) {}

    template <typename TInitializer, typename TBuffer>
    void Init(TInitializer& initializer, TBuffer& loadBuffer)
    {
        NSComposeKernel::Init<0>(initializer, loadBuffer, sublayers);
    }
        
    template <typename TSave>
    void SaveWeights(TSave& saver) const
    {
        NSComposeKernel::SaveWeights<0>(saver, sublayers);
    }
        
    template <typename TGradCollector>
    void GradCollect(TGradCollector& col)
    {
        NSComposeKernel::GradCollect<0>(col, sublayers);
    }
        
    void NeutralInvariant() const
    {
        NSComposeKernel::NeutralInvariant<0>(sublayers);
    }
    
    template <typename TIn>
    auto FeedForward(TIn&& p_in)
    {
        auto inInternal = NSComposeKernel::CreateInputInternalBuf<0, TOrderedSublayerSeq, TSublayerInstCont>(InternalResult::Create());
        auto outInternal = NSComposeKernel::CreateOutputInternalBuf<0, TOrderedSublayerSeq, TSublayerInstCont>(InternalResult::Create());

        auto inputs = NSComposeKernel::FillInput<0, typename TKernelTopo::InputConnects>(p_in, std::move(inInternal));
        auto outputs = NSComposeKernel::FeedForward<0, TOrderedSublayerSeq, typename TKernelTopo::InternalFMap>(sublayers, std::move(inputs), std::move(outInternal));
        return NSComposeKernel::FillOutput<0, typename TKernelTopo::OutputConnects>(outputs, LayerOutputCont<ComposeKernel>());
    }
    
    template <typename TGrad>
    auto FeedBackward(const TGrad& p_grad)
    {
        if constexpr ((!IsFeedbackOutput) && (!IsUpdate))
        {
            return LayerInputCont<ComposeKernel>();
        }
        else
        {
            auto inInternal = NSComposeKernel::CreateInputInternalBuf<0, TOrderedSublayerSeq, TSublayerInstCont>(InternalResult::Create());
            auto outInternal = NSComposeKernel::CreateOutputInternalBuf<0, TOrderedSublayerSeq, TSublayerInstCont>(InternalResult::Create());

            auto inputGrads = NSComposeKernel::FillInputGrad<0, typename TKernelTopo::OutputConnects>(p_grad, std::move(outInternal));
        
            auto outputs = NSComposeKernel::FeedBackward<ArraySize<SublayerArray>, TOrderedSublayerSeq,
                                                         typename TKernelTopo::InternalBMap>(sublayers, std::move(inputGrads), std::move(inInternal));
                                                     
            return NSComposeKernel::FillOutputGrad<0, typename TKernelTopo::InputConnects>(outputs, LayerInputCont<ComposeKernel>());
        }
    }
private:
    SublayerArray sublayers;
};
}