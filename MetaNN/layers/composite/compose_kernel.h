#pragma once
#include <MetaNN/facilities/cont_metafuns/_.h>
#include <MetaNN/layers/facilities/make_layer.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/layer_in_map.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/_.h>
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
            : imp<Sequential::PushBack<TS, Sublayer<TLayerName, TLayer>>,
                  TI, TO, TIt,
                  T...>
    {};
    
    template <typename TS, typename TI, typename TO, typename TIt,
              typename TInPort, typename TInLayerName, typename TInLayerPort,
              typename...T>
    struct imp<TS, TI, TO, TIt, InConnect<TInPort, TInLayerName, TInLayerPort>, T...>
            : imp<TS,
                  Sequential::PushBack<TI, InConnect<TInPort, TInLayerName, TInLayerPort>>,
                  TO, TIt,
                  T...>
    {};
    
    template <typename TS, typename TI, typename TO, typename TIt,
              typename TOutLayerName, typename TOutLayerPort, typename TOutPort,
              typename...T>
    struct imp<TS, TI, TO, TIt, OutConnect<TOutLayerName, TOutLayerPort, TOutPort>, T...>
            : imp<TS, TI,
                  Sequential::PushBack<TO, OutConnect<TOutLayerName, TOutLayerPort, TOutPort>>,
                  TIt,
                  T...>
    {};
    
    template <typename TS, typename TI, typename TO, typename TIt, 
              typename TOutLayerName, typename TOutPort, typename TInLayerName, typename TInPort,
              typename...T>
    struct imp<TS, TI, TO, TIt, 
               InternalConnect<TOutLayerName, TOutPort, TInLayerName, TInPort>, T...>
            : imp<TS, TI, TO,
                  Sequential::PushBack<TIt, InternalConnect<TOutLayerName, TOutPort, TInLayerName, TInPort>>,
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
using InternalFMap_ = MultiMap::Insert_<TState, typename TInput::OutLayer, TInput>;

template <typename TInternalClauses>
using InternalFMap = Sequential::Fold<ClauseSeq<>, TInternalClauses,
                                                   InternalFMap_>;

/// InternalBMap
template <typename TState, typename TInput>
using InternalBMap_ = MultiMap::Insert_<TState, typename TInput::InLayer, TInput>;

template <typename TInternalClauses>
using InternalBMap = Sequential::Fold<ClauseSeq<>, TInternalClauses,
                                                   InternalBMap_>;

/// InputFMap
template <typename TState, typename TInput>
using InputFMap_ = MultiMap::Insert_<TState, typename TInput::InLayerName, TInput>;

template <typename TInputClauses>
using InputFMap = Sequential::Fold<ClauseSeq<>, TInputClauses,
                                                InputFMap_>;

/// OutputBMap
template <typename TState, typename TOutput>
using OutputBMap_ = MultiMap::Insert_<TState, typename TOutput::OutLayerName, TOutput>;

template <typename TOutputClauses>
using OutputBMap = Sequential::Fold<ClauseSeq<>, TOutputClauses,
                                                 OutputBMap_>;

/// SublayerNameSet
template <typename TState, typename TInput>
using SublayerNameSet_ = Set::Insert_<TState, typename TInput::LayerName, true>;
    
template <typename TSublayerClauses>
using SublayerNameSet = Sequential::Fold<ClauseSeq<>, TSublayerClauses,
                                                      SublayerNameSet_>;

template <typename TSublayer>
struct SublayerNamePicker_
{
    using type = typename TSublayer::LayerName;
};

template <typename TSublayerClauses>
using SublayerMap = Map::CreateFromItems<TSublayerClauses, SublayerNamePicker_, ClauseSeq>;

/// InternalLayerSet
template <typename TState, typename TInput>
using InternalInLayerSet_ = Set::Insert_<TState, typename TInput::InLayer, true>;

template <typename TInternalClauses>
using InternalInLayerSet = Sequential::Fold<ClauseSeq<>, TInternalClauses,
                                                         InternalInLayerSet_>;

template <typename TState, typename TInput>
using InternalOutLayerSet_ = Set::Insert_<TState, typename TInput::OutLayer, true>;

template <typename TInternalClauses>
using InternalOutLayerSet = Sequential::Fold<ClauseSeq<>, TInternalClauses,
                                                          InternalOutLayerSet_>;

template <typename TInternalClauses>
using InternalLayerSet = Sequential::Fold<InternalInLayerSet<TInternalClauses>, TInternalClauses,
                                                       InternalOutLayerSet_>;
                                                       
/// InputNamePortSet
template <typename TState, typename TInput>
using InternalInNamePortSet_ = Set::Insert_<TState,
                                            Helper::Pair<typename TInput::InLayer,
                                            typename TInput::InPort>,
                                            true>;

template <typename TState, typename TInput>
using InputNamePortSet_ = Set::Insert_<TState, 
                                       Helper::Pair<typename TInput::InLayerName,
                                       typename TInput::InLayerPort>,
                                       true>;

template <typename TInternalClauses>
using InternalInNamePortSet = Sequential::Fold<ClauseSeq<>, TInternalClauses,
                                                            InternalInNamePortSet_>;

template <typename TInternalClauses, typename TInputClauses>
using InputNamePortSet = Sequential::Fold<InternalInNamePortSet<TInternalClauses>, TInputClauses,
                                                       InputNamePortSet_>;

/// InputLayerSet
template <typename TState, typename TInput>
using InputLayerSet_ = Set::Insert_<TState, typename TInput::InLayerName, true>;

template <typename TInputClauses>
using InputLayerSet = Sequential::Fold<ClauseSeq<>, TInputClauses,
                                                    InputLayerSet_>;
                                                       
/// OutputPortSet
template <typename TState, typename TInput>
using OutputPortSet_ = Set::Insert_<TState, typename TInput::OutPort, true>;

template <typename TOutputClauses>
using OutputPortSet = Sequential::Fold<ClauseSeq<>, TOutputClauses,
                                                    OutputPortSet_>;
                                                    
/// OutputLayerSet
template <typename TState, typename TInput>
using OutputLayerSet_ = Set::Insert_<TState, typename TInput::OutLayerName, true>;

template <typename TInternalClauses>
using OutputLayerSet = Sequential::Fold<ClauseSeq<>, TInternalClauses,
                                                     OutputLayerSet_>;
}

/// ========= Internal Tag Should In Sublayer ==========================================
template <typename TInternalClauses, typename TSublayerNameSet>
constexpr bool InternalTagInSublayer = false;

template <typename...TInterTags, typename TSublayerNameSet>
constexpr bool InternalTagInSublayer<ClauseSeq<TInterTags...>, TSublayerNameSet> =
    (Set::HasKey<TSublayerNameSet, typename TInterTags::InLayer> && ...) &&
    (Set::HasKey<TSublayerNameSet, typename TInterTags::OutLayer> && ...);

/// ========= Input Tag Should In Sublayer =============================================
template <typename TInputClauses, typename TSublayerNameSet>
constexpr bool InputTagInSubLayer = false;

template <typename...TInputTags, typename TSublayerNameSet>
constexpr bool InputTagInSubLayer<ClauseSeq<TInputTags...>, TSublayerNameSet> = 
    (Set::HasKey<TSublayerNameSet, typename TInputTags::InLayerName> && ...);

/// ========= Output Tag Should In Sublayer ============================================
template <typename TOutputClauses, typename TSublayerNameSet>
constexpr bool OutputTagInSubLayer = false;

template <typename...TOutputTags, typename TSublayerNameSet>
constexpr bool OutputTagInSubLayer<ClauseSeq<TOutputTags...>, TSublayerNameSet> = 
    (Set::HasKey<TSublayerNameSet, typename TOutputTags::OutLayerName> && ...);
    
/// ========= Sublayers Tags Sould Exist in Other Sets =================================
template <typename TSublayerNameSet,
          typename TInterLayerSet, typename TInLayerSet, typename TOutLayerSet>
constexpr bool SublayerTagInOtherArrays = false;

template <typename...TSublayerElems,
          typename TInterLayerSet, typename TInLayerSet, typename TOutLayerSet>
constexpr bool SublayerTagInOtherArrays<ClauseSeq<TSublayerElems...>,
                                        TInterLayerSet, TInLayerSet, TOutLayerSet> = 
    ((Set::HasKey<TInterLayerSet, TSublayerElems> ||
      Set::HasKey<TInLayerSet, TSublayerElems> ||
      Set::HasKey<TOutLayerSet, TSublayerElems>) && ...);

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
        static constexpr bool inInter = Set::HasKey<TInterLayers, typename TCur::LayerName>;
        
        using NewOrdered = typename std::conditional_t<inInter,
                                                       Identity_<TSublayerOrdered>,
                                                       Sequential::PushBack_<TSublayerOrdered, typename TCur::LayerName>>::type;
        using NewUnordered = typename std::conditional_t<inInter,
                                                         Sequential::PushBack_<TSubLayerUnordered, typename TCur::LayerName>,
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
        static constexpr bool inInterIn = Set::HasKey<TCheckInterSet, typename TCur::OutLayer>;
        
        using NewRemainInters = typename std::conditional_t<inInterIn,
                                                            Sequential::PushBack_<TRemainInters, TCur>,
                                                            Identity_<TRemainInters>>::type;
        using NewTagContainer = typename std::conditional_t<inInterIn,
                                                            Identity_<TPostTags>,
                                                            Set::Insert_<TPostTags, typename TCur::OutLayer, true>>::type;

        using nextStep = InternalLayerPrune<TCheckInterSet, NewRemainInters, NewTagContainer, T3...>;

        using RemainIters = typename nextStep::RemainIters;
        using PostTags = typename nextStep::PostTags;
    };
    
    template <typename TOrderedSublayers, typename TUnorderedSublayers,
              typename TCheckInternals>
    struct MainLoop
    {
        static_assert(Sequential::Size<TCheckInternals> == 0);
        using type = Sequential::Cascade<TOrderedSublayers, TUnorderedSublayers>;
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
        static_assert((Sequential::Size<NewInter> < Sequential::Size<ClauseSeq<TIC, TI...>>),
                      "Cycle exist in the compose layer");

        using NewOrdered = Sequential::Cascade<TOrderedSublayers, PostTags>;
        using NewUnordered = Sequential::Fold<TUnorderedSublayers, PostTags,
                                                           Set::Erase_>;
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
            using type = Helper::KVBinder<TSublayerName, SubPolicyPicker<TPolicy, TSublayerName>>;
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
        constexpr static size_t pos = Sequential::Order<SublayerNameSeq, TCur>;
        using OriType = Sequential::At<TInputPolicyCont, pos>;
        using NewPolicy = ChangePolicy<PFeedbackOutput, typename OriType::ValueType>;
        using NewInputPolicyCont
            = Sequential::Set<TInputPolicyCont, pos,
                                           Helper::KVBinder<typename OriType::KeyType, NewPolicy>>;
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
            constexpr static bool ShouldUpdate = Set::HasKey<TUpdateLayerSet, LayerName>;
            using NewPolicy = typename std::conditional_t<ShouldUpdate,
                                                          ChangePolicy_<PFeedbackOutput, OriPolicy>,
                                                          Identity_<OriPolicy>>::type;
            using type = Helper::KVBinder<LayerName, NewPolicy>;
        };
    };

    template <typename TOutLayerName, typename TSublayerPolicies, typename TInterMap>
    struct UpdatePolicyThroughInterMap_
    {
        using LayerProcs = MultiMap::Find<TInterMap, TOutLayerName>;
        using UpdateLayerSet = Set::CreateFromItems<LayerProcs, PickInLayerFromInternalConnect_, true>;
        using NewSublayerPolicies
            = Sequential::Transform<TSublayerPolicies,
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

        using NewProcessed = Sequential::PushBack<TProcessed, TCur>;

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
    struct InputTypeFillInConnectHelper_<TOutputCont, TDataMap, Helper::ValueSequence<TCur, TRemain...>>
    {
        using FromPort = typename TCur::InPort;
        using ToPort = typename TCur::InLayerPort;
        using AimType = typename TDataMap::template Find<FromPort>;
        using NewCont = Sequential::PushBack<TOutputCont, LayerKV<ToPort, AimType>>;
        using type = typename InputTypeFillInConnectHelper_<NewCont, TDataMap, Helper::ValueSequence<TRemain...>>::type;
    };
    
    template <typename TSublayerCont, typename TInFMap, typename TOutputCont, typename TInputDataMap>
    struct InputTypeFillInConnect_
    {
        using type = TOutputCont;
    };
    
    template <typename TInFMap, typename TOutputCont, typename TInputDataMap, typename TCur, typename... TRemain>
    struct InputTypeFillInConnect_<ClauseSeq<TCur, TRemain...>, TInFMap, TOutputCont, TInputDataMap>
    {
        using CurValueSeq = MultiMap::Find<TInFMap, TCur>;
        using TCurDataMap = typename InputTypeFillInConnectHelper_<LayerInMap<>, TInputDataMap, CurValueSeq>::type;
        using NewOutputCont = Sequential::PushBack<TOutputCont, TCurDataMap>;
        using type = typename InputTypeFillInConnect_<ClauseSeq<TRemain...>, TInFMap, NewOutputCont, TInputDataMap>::type;
    };

    template <typename TSublayerOutputMap, typename TConnectInfo, typename SublayerSeq, typename TOutputCont>
    struct InputTypeFillInternalConnectHelper2_
    {
        using type = TOutputCont;
    };
    
    template <typename TCur, typename... TRemain, typename TSublayerOutputMap, typename SublayerSeq, typename TOutputCont>
    struct InputTypeFillInternalConnectHelper2_<TSublayerOutputMap, Helper::ValueSequence<TCur, TRemain...>,
                                                SublayerSeq, TOutputCont>
    {
        using InLayer = typename TCur::InLayer;
        constexpr static size_t pos = Sequential::Order<SublayerSeq, InLayer>;
        using OriOutput = Sequential::At<TOutputCont, pos>;
        
        using OutPort = typename TCur::OutPort;
        using AimType = typename TSublayerOutputMap::template Find<OutPort>;
        
        using InPort = typename TCur::InPort;
        static_assert(std::is_same_v<typename OriOutput::template Find<InPort>, NullParameter>);
        
        using NewOutput = Sequential::PushBack<OriOutput, LayerKV<InPort, AimType>>;
        using NewOutputCont = Sequential::Set<TOutputCont, pos, NewOutput>;
        
        using type
            = typename InputTypeFillInternalConnectHelper2_<TSublayerOutputMap, Helper::ValueSequence<TRemain...>,
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
        using TCurLayer = Sequential::At<TSublayerNameCont, CurID>;
        using InternalConnections = MultiMap::Find<TInternalMap, TCurLayer>;
        
        using NewInputMapCont
            = typename std::conditional_t<(Sequential::Size<InternalConnections> == 0),
                                          Identity_<TInputMapCont>,
                                          InputTypeFillInternalConnectHelper_<Map::Find<TSublayerMap, TCurLayer>,
                                                                              Sequential::At<TInputMapCont, CurID>,
                                                                              Map::Find<TPolicyCont, TCurLayer>,
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
        using TLayerTemp = Map::Find<TLayerMap, TCurLayer>;
        static_assert(!std::is_same_v<TLayerTemp, void>);
        using TCurPolicy = Map::Find<TPolicies, TCurLayer>;
        static_assert(!std::is_same_v<TCurPolicy, void>);
        using LayerType = typename TLayerTemp::template LayerType<TInputHead, TCurPolicy>;
        using NewOutCont = Sequential::PushBack<TOutCont, LayerType>;
        using type = typename Instantiation_<NewOutCont, ClauseSeq<TRemainLayers...>, TLayerMap, std::tuple<TRemainInputMap...>,
                                             TPolicies>::type;
    };
    
    template <typename TInputs, typename OrderedSublayers, typename TSublayerClauses, typename InConnects, typename InterConnects, typename SublayerPolicyFinal>
    struct NontrivialInst_
    {
        //  Fill Input type container with in-connections
        using InputTypeCont1 = typename NSSI::InputTypeFillInConnect_<OrderedSublayers,
                                                                      ClauseRefine::InputFMap<InConnects>,
                                                                      std::tuple<>, TInputs>::type;
        // Fill Input type container with inter-connections
        using InputTypeContFinal = typename NSSI::InputTypeFillInternalConnect_<0, Sequential::Size<OrderedSublayers>,
                                                                                OrderedSublayers,
                                                                                InputTypeCont1, SublayerPolicyFinal,
                                                                                ClauseRefine::SublayerMap<TSublayerClauses>,
                                                                                ClauseRefine::InternalFMap<InterConnects>>::type;
        /// Instantiation
        using type = typename NSSI::Instantiation_<std::tuple<>, OrderedSublayers, ClauseRefine::SublayerMap<TSublayerClauses>,
                                                   InputTypeContFinal, SublayerPolicyFinal>::type;
    };
    
    template <typename TSublayerMap, typename SublayerPolicyFinal>
    struct TrivialInstHelper_
    {
        template <typename TCurLayer>
        struct apply
        {
            using SublayerInfo = Map::Find<TSublayerMap, TCurLayer>;
            using SublayerPolicy = Map::Find<SublayerPolicyFinal, TCurLayer>;
            using type = MakeInferLayer<SublayerInfo::template LayerType, SublayerPolicy>;
        };
    };
    
    template <typename OrderedSublayers, typename TSublayerClauses, typename SublayerPolicyFinal>
    struct TrivialInst_
    {
        using type = Sequential::Transform<OrderedSublayers,
                                                        TrivialInstHelper_<ClauseRefine::SublayerMap<TSublayerClauses>,
                                                                          SublayerPolicyFinal>::template apply,
                                                        std::tuple>;
    };
}

template <typename TInputs, typename TPolicies,
          typename OrderedSublayers, typename TSublayerClauses, typename InConnects, typename InterConnects, typename OutConnects>
struct SublayerInstantiation_
{
    static_assert(IsPolicyContainer<TPolicies>, "Not a Policy Container");
    
    using SublayerPolicy1 = Sequential::Transform<OrderedSublayers,
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
    using type = typename std::conditional_t<IsEmptyLayerInMap<TInputs>,
                                             NSSI::TrivialInst_<OrderedSublayers, TSublayerClauses, SublayerPolicyFinal>,
                                             NSSI::NontrivialInst_<TInputs, OrderedSublayers, TSublayerClauses, InConnects, InterConnects, SublayerPolicyFinal>>::type;

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
    using InternalBMap = NSComposeKernel::ClauseRefine::InternalBMap<InterConnects>;
    
/// ========== Asserts =================================================
    static_assert((Sequential::Size<Sublayers> != 0), "Sublayer is empty.");
    static_assert((Sequential::Size<Sublayers> == Sequential::Size<SublayerNameSet>), "Two or more sublayers have same tag.");
    static_assert(Sequential::Size<InputConnects> + Sequential::Size<InterConnects> ==
                  Sequential::Size<NSComposeKernel::ClauseRefine::InputNamePortSet<InterConnects, InputConnects>>,
                  "One input corresponds to two or more sources.");
    static_assert(Sequential::Size<OutputConnects> == Sequential::Size<NSComposeKernel::ClauseRefine::OutputPortSet<OutputConnects>>,
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
        static_assert(Sequential::Size<TSublayerNameTuple> == Sequential::Size<TSublayerTuple>);
    public:
        using SublayerArray
            = Sequential::Transform<TSublayerTuple, NSComposeKernel::AddSharedPtrWrapper_, std::tuple>;

    private:
        template <size_t N>
        void FillGap()
        {
            if constexpr (N == Sequential::Size<TSublayerTuple>)
            {
                return;
            }
            else
            {
                using AimType = typename Sequential::At<TSublayerTuple, N>;
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
            constexpr static size_t Pos = Sequential::Order<TSublayerNameTuple, TLayerName>;
            
            using AimType = typename Sequential::At<TSublayerTuple, Pos>;
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
        if constexpr (N != Sequential::Size<TSublayers>)
        {
            auto& layer = std::get<N>(sublayers);
            LayerInit(*layer, initializer, loadBuffer);
            Init<N + 1>(initializer, loadBuffer, sublayers);
        }
    }

    
    template <size_t N, typename TSave, typename TSublayers>
    void SaveWeights(TSave& saver, const TSublayers& sublayers)
    {
        if constexpr (N != Sequential::Size<TSublayers>)
        {
            auto& layer = std::get<N>(sublayers);
            LayerSaveWeights(*layer, saver);
            SaveWeights<N + 1>(saver, sublayers);
        }
    }
    
    template <size_t N, typename TGradCollector, typename TSublayers>
    void GradCollect(TGradCollector& collector, TSublayers& sublayers)
    {
        if constexpr (N != Sequential::Size<TSublayers>)
        {
            auto& layer = std::get<N>(sublayers);
            LayerGradCollect(*layer, collector);
            GradCollect<N + 1>(collector, sublayers);
        }
    }

    template <size_t N, typename TSublayers>
    void NeutralInvariant(TSublayers& sublayers)
    {
        if constexpr (N != Sequential::Size<TSublayers>)
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
        if constexpr (N == Sequential::Size<TLayerInst>)
        {
            return std::forward<TInput>(m_input);
        }
        else
        {
            using TCurName = Sequential::At<TLayerNames, N>;
            using TCurInst = Sequential::At<TLayerInst, N>;
            auto inputCont = LayerInputCont<TCurInst>();
            auto newInput = std::move(m_input).template Set<TCurName>(std::move(inputCont));
            return CreateInputInternalBuf<N + 1, TLayerNames, TLayerInst>(std::move(newInput));
        }
    }
    
    template <size_t N, typename TLayerNames, typename TLayerInst, typename TOutput>
    auto CreateOutputInternalBuf(TOutput&& m_output)
    {
        if constexpr (N == Sequential::Size<TLayerInst>)
        {
            return std::forward<TOutput>(m_output);
        }
        else
        {
            using TCurName = Sequential::At<TLayerNames, N>;
            using TCurInst = Sequential::At<TLayerInst, N>;
            auto outputCont = LayerOutputCont<TCurInst>();
            auto newOutput = std::move(m_output).template Set<TCurName>(std::move(outputCont));
            return CreateOutputInternalBuf<N + 1, TLayerNames, TLayerInst>(std::move(newOutput));
        }
    }
    
    template <size_t N, typename TInputClauses, typename TIn, typename TInternal>
    auto FillInput(const TIn& p_in, TInternal&& p_internal)
    {
        if constexpr (N == Sequential::Size<TInputClauses>)
        {
            return std::move(p_internal);
        }
        else
        {
            using TCur = Sequential::At<TInputClauses, N>;
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
        if constexpr (Sequential::Size<TMap> == 0)
        {
            return std::move(p_aim);
        }
        else 
        {
            if constexpr (N == Sequential::Size<TMap>)
            {
                return std::move(p_aim);
            }
            else
            {
                using TCur = Sequential::At<TMap, N>;
                
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
        if constexpr (N == Sequential::Size<TLayerInfo>)
        {
            return std::move(m_output);
        }
        else
        {
            using TCurLayerName = Sequential::At<TLayerInfo, N>;
            auto source = std::forward<TInput>(p_input).template Get<TCurLayerName>();
            auto forwardRes = std::get<N>(sublayers)->FeedForward(std::move(source));
            
            using ItemsFromMap = MultiMap::Find<TFMap, TCurLayerName>;
            
            auto newInput = ForwardFillInternal<0, ItemsFromMap>(forwardRes, std::move(p_input));
            auto newOutput = std::move(m_output).template Set<TCurLayerName>(forwardRes);
            
            return FeedForward<N+1, TLayerInfo, TFMap>(sublayers, std::move(newInput), std::move(newOutput));
        }
    }
    
    template <size_t N, typename TOutputClauses, typename TIn, typename TAim>
    auto FillOutput(const TIn& p_in, TAim&& p_aim)
    {
        if constexpr (N == Sequential::Size<TOutputClauses>)
        {
            return std::move(p_aim);
        }
        else
        {
            using TCur = Sequential::At<TOutputClauses, N>;
            
            auto sourceLayer = p_in.template Get<typename TCur::OutLayerName>();
            auto source = sourceLayer.template Get<typename TCur::OutLayerPort>();
            
            auto newAim = std::move(p_aim).template Set<typename TCur::OutPort>(std::move(source));
            return FillOutput<N + 1, TOutputClauses>(p_in, std::move(newAim));
        }
    }
    
    template <size_t N, typename TOutputClauses, typename TInGrad, typename TInternal>
    auto FillInputGrad(const TInGrad& p_inGrad, TInternal&& p_internal)
    {
        if constexpr (N == Sequential::Size<TOutputClauses>)
        {
            return std::move(p_internal);
        }
        else
        {
            using TCur = Sequential::At<TOutputClauses, N>;
            
            auto source = p_inGrad.template Get<typename TCur::OutPort>();
            auto dest = p_internal.template Get<typename TCur::OutLayerName>();
            if constexpr (decltype(dest)::template IsValueEmpty<typename TCur::OutLayerPort>)
            {
                auto fillRes = std::move(dest).template Set<typename TCur::OutLayerPort>(std::move(source));
                auto newInternal = std::move(p_internal).template Set<typename TCur::OutLayerName>(std::move(fillRes));
                return FillInputGrad<N + 1, TOutputClauses>(p_inGrad, std::move(newInternal));
            }
            else
            {
                const auto& prevInfo = dest.template Get<typename TCur::OutLayerPort>();
                auto fillRes = std::move(dest).template Set<typename TCur::OutLayerPort>(prevInfo + source);
                auto newInternal = std::move(p_internal).template Set<typename TCur::OutLayerName>(std::move(fillRes));
                return FillInputGrad<N + 1, TOutputClauses>(p_inGrad, std::move(newInternal));
            }
        }
    }
    
    template <size_t N, typename TMap, typename TBackwardRes, typename TAim>
    auto BackwardFillInternal(const TBackwardRes& input, TAim&& p_aim)
    {
        if constexpr (Sequential::Size<TMap> == 0)
        {
            return std::move(p_aim);
        }
        else 
        {
            if constexpr (N == Sequential::Size<TMap>)
            {
                return std::move(p_aim);
            }
            else
            {
                using TCur = Sequential::At<TMap, N>;
                
                auto des = std::move(p_aim).template Get<typename TCur::OutLayer>();
                if constexpr (decltype(des)::template IsValueEmpty<typename TCur::OutPort>)
                {
                    if constexpr (!TBackwardRes::template IsValueEmpty<typename TCur::InPort>)
                    {
                        auto value = input.template Get<typename TCur::InPort>();
                        auto newDes = std::move(des).template Set<typename TCur::OutPort>(std::move(value));
                        auto newAim = std::move(p_aim).template Set<typename TCur::OutLayer>(std::move(newDes));
                        return BackwardFillInternal<N + 1, TMap>(input, std::move(newAim));
                    }
                    else
                    {
                        return BackwardFillInternal<N + 1, TMap>(input, std::forward<TAim>(p_aim));
                    }
                }
                else
                {
                    const auto& value = input.template Get<typename TCur::InPort>();
                    const auto& prevInfo = std::move(des).template Get<typename TCur::OutPort>();
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
            using TCurLayerName = Sequential::At<TLayerInfo, N - 1>;
            auto source = std::move(p_input).template Get<TCurLayerName>();
            auto backwardRes = std::get<N - 1>(sublayers)->FeedBackward(std::move(source));
            
            using ItemsFromMap = MultiMap::Find<TBMap, TCurLayerName>;
            
            auto newInput = BackwardFillInternal<0, ItemsFromMap>(backwardRes, std::move(p_input));
            auto newOutput = std::move(m_output).template Set<TCurLayerName>(backwardRes);
            
            return FeedBackward<N-1, TLayerInfo, TBMap>(sublayers, std::move(newInput), std::move(newOutput));
        }
    }
    
    template <size_t N, typename TInputClauses, typename TInGrad, typename TAim>
    auto FillOutputGrad(const TInGrad& p_in, TAim&& p_aim)
    {
        if constexpr (N == Sequential::Size<TInputClauses>)
        {
            return std::move(p_aim);
        }
        else
        {
            using TCur = Sequential::At<TInputClauses, N>;
            
            auto sourceLayer = p_in.template Get<typename TCur::InLayerName>();
            auto source = sourceLayer.template Get<typename TCur::InLayerPort>();
            
            if constexpr (RemConstRef<TAim>::template IsValueEmpty<typename TCur::InPort>)
            {
                auto newAim = std::move(p_aim).template Set<typename TCur::InPort>(std::move(source));
                return FillOutputGrad<N + 1, TInputClauses>(p_in, std::move(newAim));
            }
            else
            {
                const auto& prevInfo = std::move(p_aim).template Get<typename TCur::InPort>();
                auto newAim = std::move(p_aim).template Set<typename TCur::InPort>(prevInfo + source);
                return FillOutputGrad<N + 1, TInputClauses>(p_in, std::move(newAim));
            }
        }
    }
};

template <typename TInputPortSet, typename TOutputPortSet, typename TInputMap, typename TPolicyCont, typename TKernelTopo>
class ComposeKernel
{
    static_assert(IsPolicyContainer<TPolicyCont>, "Parameter is not a policy container.");
    using PlainPolicies = PlainPolicy<TPolicyCont>;
    
public:
    using InputPortSet = TInputPortSet;
    using OutputPortSet = TOutputPortSet;
    using InputMap = typename std::conditional_t<std::is_same_v<TInputMap, NullParameter>,
                                                 EmptyLayerInMap_<InputPortSet>,
                                                 Identity_<TInputMap>>::type;
    static_assert(CheckInputMapAvailable_<InputMap, InputPortSet>::value);
    
private:
    using TOrderedSublayerSeq = typename TKernelTopo::TopologicalOrdering;
    using TSublayerInstCont = typename TKernelTopo::template Instances<InputMap, TPolicyCont>;
    using SublayerArray = typename NSComposeKernel::SublayerArrayMaker<TOrderedSublayerSeq, TSublayerInstCont>::SublayerArray;
    using InternalResult = NSComposeKernel::InternalResult<TOrderedSublayerSeq>;

public:
    template <typename TSublayerName>
    using SublayerType = typename Sequential::At<TSublayerInstCont,
                                                              Sequential::Order<TOrderedSublayerSeq, TSublayerName>>;

    static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, PlainPolicies>::IsFeedbackOutput;
    static constexpr bool IsUpdate = NSComposeKernel::IsComposeLayerUpdate_<TSublayerInstCont>::value;

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
        
            auto outputs = NSComposeKernel::FeedBackward<Sequential::Size<SublayerArray>, TOrderedSublayerSeq, typename TKernelTopo::InternalBMap>(sublayers, std::move(inputGrads), std::move(inInternal));
            if constexpr (IsFeedbackOutput)
            {
                return NSComposeKernel::FillOutputGrad<0, typename TKernelTopo::InputConnects>(outputs, LayerInputCont<ComposeKernel>());
            }
            else
            {
                return LayerInputCont<ComposeKernel>();
            }
        }
    }
private:
    SublayerArray sublayers;
};
}