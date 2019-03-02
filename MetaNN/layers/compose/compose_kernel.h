#pragma once
#include <MetaNN/policies/change_policy.h>
#include <MetaNN/facilities/cont_metafuns/sequential.h>
#include <MetaNN/facilities/cont_metafuns/multi_map.h>
#include <MetaNN/facilities/cont_metafuns/set.h>
#include <type_traits>

namespace MetaNN
{
template <typename TLayerName, template<typename> class TLayer>
struct Sublayer
{
    using LayerName = TLayerName;

    template <typename T> using LayerType = TLayer<T>;
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
              typename TLayerName, template<typename> class TLayer, typename...T>
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

/// SublayerNameSet
template <typename TState, typename TInput>
using SublayerNameSet_ = ContMetaFun::Set::Insert_<TState, typename TInput::LayerName, true>;
    
template <typename TSublayerClauses>
using SublayerNameSet = ContMetaFun::Sequential::Fold<ClauseSeq<>, TSublayerClauses,
                                                      SublayerNameSet_>;

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
    
    template <typename TSublayerClause>
    struct LayerName2Info;
    
    template <typename ... TSublayer>
    struct LayerName2Info<ClauseSeq<TSublayer...>>
        : public ContMetaFun::Helper::KVBinder<typename TSublayer::LayerName, TSublayer>...
    {
        using ContMetaFun::Helper::KVBinder<typename TSublayer::LayerName, TSublayer>::apply...;
    };
    
    template <typename TSublayerClause>
    struct SeqLayerName2Info_
    {
        template <typename TState, typename TInput>
        using apply = ContMetaFun::Sequential::PushBack_<TState,
                                                         decltype(LayerName2Info<TSublayerClause>::apply((TInput*)nullptr))>;
    };
}

template <typename TSublayerClause, typename TInterClause>
struct TopologicalOrdering_;

template <typename...TSublayers, typename TInterClause>
struct TopologicalOrdering_<ClauseSeq<TSublayers...>, TInterClause>
{
    using SublayerPreRes = NSTPO::SublayerPreprocess_<ClauseRefine::InternalLayerSet<TInterClause>,
                                                      ClauseSeq<>, ClauseSeq<>, TSublayers...>;

    using SublayerTags = typename NSTPO::MainLoop<typename SublayerPreRes::Ordered,
                                                  typename SublayerPreRes::Unordered,
                                                  TInterClause>::type;
                                                  
    using type = ContMetaFun::Sequential::Fold<ClauseSeq<>, SublayerTags,
                                               NSTPO::SeqLayerName2Info_<ClauseSeq<TSublayers...>>::template apply>;
};

/// ========= Sublayer Instantiation ===================================================
namespace NSSI
{
    template <typename TSublayer, typename TPolicy>
    struct SublayerWithPolicy
    {
        using Sublayer = TSublayer;
        using Policy = TPolicy;
    };
    
    template <typename... TInst> struct SublayerPolicyCont;
    
    template <typename TPolicyCont>
    struct GetSublayerPolicy
    {
        template <typename TState, typename TInput>
        struct apply
        {
            using InputPolicy = SubPolicyPicker<TPolicyCont, typename TInput::LayerName>;
            using type = ContMetaFun::Sequential::PushBack<TState, SublayerWithPolicy<TInput, InputPolicy>>;
        };
    };
    
    template <typename TInst, typename InConnectSet>
    struct FbSetByInConnectionHelper_
    {
        using type = typename std::conditional_t<ContMetaFun::Set::HasKey<InConnectSet, typename TInst::Sublayer>,
                                                 ChangePolicy_<PFeedbackOutput, typename TInst::Policy>,
                                                 Identity_<typename TInst::Policy>>::type;
    };
    
    template <typename TInstCont, typename InConnectSet>
    struct FbSetByInConnection_;
    
    template <typename... TInst, typename InConnectSet>
    struct FbSetByInConnection_<SublayerPolicyCont<TInst...>, InConnectSet>
    {
        using type = SublayerPolicyCont<typename FbSetByInConnectionHelper_<TInst, InConnectSet>::type ...>;
    };
    
    template <typename TItem, typename TQuery>
    constexpr bool HasInternalIn = false;
    
    template <typename TKey, typename...TValue, typename TQuery>
    constexpr bool HasInternalIn<ContMetaFun::Helper::Pair<TKey, TValue...>, TQuery> = (std::is_same_v<typename TValue::InLayer, TQuery> || ...);
    
    template <typename TInItems>
    struct FbOutModHelper_
    {
        template <typename TState, typename TInput>
        struct apply
        {
            constexpr static bool modify = HasInternalIn<TInItems, typename TInput::Sublayer::LayerName>;
            using NewPolicy = typename std::conditional_t<modify,
                                                          ChangePolicy_<PFeedbackOutput, typename TInput::Policy>,
                                                          Identity_<typename TInput::Policy>>::type;
            using NewItem = SublayerWithPolicy<typename TInput::Sublayer, NewPolicy>;
            using type = ContMetaFun::Sequential::PushBack<TState, NewItem>;
        };
    };
    
    template <typename TProcessed, typename TRemain, typename InterFMap>
    struct FeedbackOutModif
    {
        using type = TProcessed;
    };

    template <typename TProcessed, typename TCur, typename...TInstElements, typename InterFMap>
    struct FeedbackOutModif<TProcessed, SublayerPolicyCont<TCur, TInstElements...>, InterFMap>
    {
        constexpr static bool isUpdate = PolicySelect<GradPolicy, typename TCur::Policy>::IsFeedbackOutput || \
                                         PolicySelect<GradPolicy, typename TCur::Policy>::IsUpdate;
                                         
        using NewProcessed = ContMetaFun::Sequential::PushBack<TProcessed, TCur>;

        using NewRemain = typename std::conditional_t<isUpdate && (sizeof...(TInstElements) != 0),
                                                      ContMetaFun::Sequential::Fold_<SublayerPolicyCont<>, SublayerPolicyCont<TInstElements...>,
                                                                                     FbOutModHelper_<ContMetaFun::MultiMap::Find<InterFMap, typename TCur::Sublayer::LayerName>>::template apply>,
                                                      Identity_<SublayerPolicyCont<TInstElements...>>>::type;
        using type = typename FeedbackOutModif<NewProcessed, NewRemain, InterFMap>::type;
    };
    
    template <typename TState, typename TInput>
    struct InstHelper_
    {
        template <typename T>
        using Layer = typename TInput::Sublayer::template LayerType<T>;
        
        using Inst = InstantiatedSublayer<typename TInput::Sublayer::LayerName, Layer<typename TInput::Policy>>;
        using type = ContMetaFun::Sequential::PushBack<TState, Inst>;
    };
}

template <typename TPolicyCont, typename OrderedSublayers, typename InConnects, typename InterConnects>
struct SublayerInstantiation
{
    static_assert(IsPolicyContainer<TPolicyCont>, "Not a Policy Container");
    
    using SublayerWithPolicyRes = ContMetaFun::Sequential::Fold<NSSI::SublayerPolicyCont<>, OrderedSublayers,
                                                                NSSI::GetSublayerPolicy<TPolicyCont>::template apply>;
                                                                
    /// if feedbackout is set in parent layer, then each sublayer that includes in InConnects should also set it to true
    constexpr static bool IsPlainPolicyFeedbackOut = PolicySelect<GradPolicy, PlainPolicy<TPolicyCont>>::IsFeedbackOutput;
    using SublayerPolicyModif1
        =typename std::conditional_t<!IsPlainPolicyFeedbackOut,
                                      Identity_<SublayerWithPolicyRes>,
                                      NSSI::FbSetByInConnection_<SublayerWithPolicyRes,
                                                                 ClauseRefine::InputLayerSet<InConnects>>>::type;
                                         
    /// for any instance A, if there is a connection A->B and A is feedbackin, then B should set to feedbackout
    using SublayerPolicyModif2 = typename NSSI::FeedbackOutModif<NSSI::SublayerPolicyCont<>,
                                                                 SublayerPolicyModif1,
                                                                 ClauseRefine::InternalFMap<InterConnects>>::type;


    /// Instantiation
    using type = ContMetaFun::Sequential::Fold<std::tuple<>, SublayerPolicyModif2,
                                               NSSI::InstHelper_>;
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
    static_assert((ArraySize<Sublayers> != 0), "Sublayer is empty.");
    static_assert((ArraySize<Sublayers> == ArraySize<SublayerNameSet>),
                  "Two or more sublayers have same tag.");
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

    template <typename TPolicyCont>
    using Instances = typename NSComposeKernel::SublayerInstantiation<TPolicyCont, TopologicalOrdering, InputConnects, InterConnects>::type;
};


namespace NSComposeKernel
{
    template <typename TState, typename TInput>
    using SublayerTypePicker_ = ContMetaFun::Sequential::PushBack_<TState, std::shared_ptr<typename TInput::LayerType>>;
    
    template <typename TLayers, typename TIndex>
    struct MapLayerName2ID;
    
    template <typename... TLayer, int... TIndex>
    struct MapLayerName2ID<std::tuple<TLayer...>, ContMetaFun::Helper::IndexSequence<TIndex...>>
        : public ContMetaFun::Helper::KVBinder<typename TLayer::LayerName, ContMetaFun::Helper::Int_<TIndex>>...
    {
        using ContMetaFun::Helper::KVBinder<typename TLayer::LayerName, ContMetaFun::Helper::Int_<TIndex>>::apply...;
    };
    
    template <typename TSublayerTuple>
    struct SublayerArrayMaker
    {
    public:
        using SublayerArray = ContMetaFun::Sequential::Fold<std::tuple<>, TSublayerTuple,
                                                            SublayerTypePicker_>;

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
                using AimType = typename ContMetaFun::Sequential::At<TSublayerTuple, N>::LayerType;
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
            constexpr static size_t ArrayLength = ArraySize<TSublayerTuple>;
            
            constexpr static size_t Pos
                = decltype(MapLayerName2ID<TSublayerTuple, ContMetaFun::Helper::MakeIndexSequence<ArrayLength>>::apply((TLayerName*)nullptr))::value;
            static_assert(Pos < ArrayLength);
            
            using AimType = typename ContMetaFun::Sequential::At<TSublayerTuple, Pos>::LayerType;
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
    
    template <typename TLayerInsts>
    struct IsComposeLayerUpdate_;
    
    template <typename... TLayerInst>
    struct IsComposeLayerUpdate_<std::tuple<TLayerInst...>>
    {
        constexpr static bool value = (TLayerInst::LayerType::IsUpdate || ...);
    };
    
    template <typename TLayerInsts>
    constexpr bool IsComposeLayerUpdate = IsComposeLayerUpdate_<TLayerInsts>::value;
    
    template <size_t N, typename TInitPolicies, typename TSublayerInfo,
              typename TInitializer, typename TBuffer, typename TSublayers>
    void Init(TInitializer& initializer, TBuffer& loadBuffer, std::ostream* log, TSublayers& sublayers)
    {
        if constexpr (N != ArraySize<TSublayers>)
        {
            auto& layer = std::get<N>(sublayers);
            using LayerInfo = typename std::tuple_element<N, TSublayerInfo>::type;
            using NewInitPolicy = SubPolicyPicker<TInitPolicies, typename LayerInfo::LayerName>;
            LayerInit<typename LayerInfo::LayerType, TInitializer, TBuffer, NewInitPolicy>(*layer, initializer, loadBuffer);
            Init<N + 1, TInitPolicies, TSublayerInfo>(initializer, loadBuffer, log, sublayers);
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
    
    template <typename TSublayers>
    struct InternalResult;
    
    template <typename... TSublayers>
    struct InternalResult<NSComposeKernel::ClauseSeq<TSublayers...>>
        : VarTypeDict<TSublayers...>
    {};

    template <size_t N, typename TLayerInst, typename TInput>
    auto CreateInputInternalBuf(TInput&& m_input)
    {
        if constexpr (N == ArraySize<TLayerInst>)
        {
            return std::forward<TInput>(m_input);
        }
        else
        {
            using TCur = ContMetaFun::Sequential::At<TLayerInst, N>;
            auto inputCont = TCur::LayerType::InputType::Create();
            auto newInput = std::move(m_input).template Set<typename TCur::LayerName>(std::move(inputCont));
            return CreateInputInternalBuf<N + 1, TLayerInst>(std::move(newInput));
        }
    }
    
    template <size_t N, typename TLayerInst, typename TInput>
    auto CreateOutputInternalBuf(TInput&& m_input)
    {
        if constexpr (N == ArraySize<TLayerInst>)
        {
            return std::forward<TInput>(m_input);
        }
        else
        {
            using TCur = ContMetaFun::Sequential::At<TLayerInst, N>;
            auto inputCont = TCur::LayerType::OutputType::Create();
            auto newInput = std::move(m_input).template Set<typename TCur::LayerName>(std::move(inputCont));
            return CreateOutputInternalBuf<N + 1, TLayerInst>(std::move(newInput));
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
        if constexpr (std::is_same_v<TMap, void>)
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
            using TCur = ContMetaFun::Sequential::At<TLayerInfo, N>;
            auto source = std::forward<TInput>(p_input).template Get<typename TCur::LayerName>();
            auto forwardRes = std::get<N>(sublayers)->FeedForward(std::move(source));
            
            using ItemsFromMap = ContMetaFun::MultiMap::Find<TFMap, typename TCur::LayerName>;
            
            // 1 to omit the key in pair
            auto newInput = ForwardFillInternal<1, ItemsFromMap>(forwardRes, std::move(p_input));
            auto newOutput = std::move(m_output).template Set<typename TCur::LayerName>(forwardRes);
            
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
                auto fillRes = std::move(dest).template Set<typename TCur::OutLayerPort>(source + prevInfo);
                auto newInternal = std::move(p_internal).template Set<typename TCur::OutLayerName>(std::move(fillRes));
                return FillInputGrad<N + 1, TOutputClauses>(p_inGrad, std::move(newInternal));
            }
        }
    }
    
    template <size_t N, typename TMap, typename TBackwardRes, typename TAim>
    auto BackwardFillInternal(const TBackwardRes& input, TAim&& p_aim)
    {
        if constexpr (std::is_same_v<TMap, void>)
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
                    auto newDes = std::move(des).template Set<typename TCur::OutPort>(value + prevInfo);
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
            
            // 1 to omit the key in pair
            auto newInput = BackwardFillInternal<1, ItemsFromMap>(backwardRes, std::move(p_input));
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
                auto newAim = std::move(p_aim).template Set<typename TCur::InPort>(source + prevInfo);
                return FillOutputGrad<N + 1, TInputClauses>(p_in, std::move(newAim));
            }
        }
    }
}

template <typename TInputType, typename TOutputType, typename TPolicyCont, typename TKernelTopo>
class ComposeKernel
{
    static_assert(IsPolicyContainer<TPolicyCont>, "Parameter is not a policy container.");
    using PlainPolicies = PlainPolicy<TPolicyCont>;
    
private:
    using TInstContainer = typename TKernelTopo::template Instances<TPolicyCont>;
    using SublayerArray = typename NSComposeKernel::SublayerArrayMaker<TInstContainer>::SublayerArray;
    using InternalResult = NSComposeKernel::InternalResult<typename TKernelTopo::SublayerNameSet>;
    
public:
    static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, PlainPolicies>::IsFeedbackOutput;
    static constexpr bool IsUpdate = NSComposeKernel::IsComposeLayerUpdate<TInstContainer>;

    using InputType = TInputType;
    using OutputType = TOutputType;

public:
    static auto CreateSubLayers()
    {
        return NSComposeKernel::SublayerArrayMaker<TInstContainer>();
    }
    
public:
    ComposeKernel(SublayerArray p_sublayers)
        : sublayers(std::move(p_sublayers)) {}
        
public:
    template <typename TInitializer, typename TBuffer,
              typename TInitPolicies = typename TInitializer::PolicyCont>
    void Init(TInitializer& initializer, TBuffer& loadBuffer, std::ostream* log = nullptr)
    {
        NSComposeKernel::Init<0, TInitPolicies, TInstContainer>(initializer, loadBuffer, log, sublayers);
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

    void NeutralInvariant()
    {
        NSComposeKernel::NeutralInvariant<0>(sublayers);
    }
    
    template <typename TIn>
    auto FeedForward(const TIn& p_in)
    {
        auto inInternal = NSComposeKernel::CreateInputInternalBuf<0, TInstContainer>(InternalResult::Create());
        auto outInternal = NSComposeKernel::CreateOutputInternalBuf<0, TInstContainer>(InternalResult::Create());
        
        auto inputs = NSComposeKernel::FillInput<0, typename TKernelTopo::InputConnects>(p_in, std::move(inInternal));
        
        auto outputs = NSComposeKernel::FeedForward<0, TInstContainer, typename TKernelTopo::InternalFMap>(sublayers, std::move(inputs), std::move(outInternal));
        
        return NSComposeKernel::FillOutput<0, typename TKernelTopo::OutputConnects>(outputs, OutputType::Create());
    }
    
    template <typename TGrad>
    auto FeedBackward(const TGrad& p_grad)
    {
        if constexpr ((!IsFeedbackOutput) && (!IsUpdate))
        {
            return OutputType::Create();
        }
        else
        {
            auto inInternal = NSComposeKernel::CreateInputInternalBuf<0, TInstContainer>(InternalResult::Create());
            auto outInternal = NSComposeKernel::CreateOutputInternalBuf<0, TInstContainer>(InternalResult::Create());
        
            auto inputGrads = NSComposeKernel::FillInputGrad<0, typename TKernelTopo::OutputConnects>(p_grad, std::move(outInternal));
        
            auto outputs = NSComposeKernel::FeedBackward<ArraySize<SublayerArray>, TInstContainer,
                                                         typename TKernelTopo::InternalBMap>(sublayers, std::move(inputGrads), std::move(inInternal));
                                                     
            return NSComposeKernel::FillOutputGrad<0, typename TKernelTopo::InputConnects>(outputs, InputType::Create());
        }
    }
    
private:
    SublayerArray sublayers;
};

template <template<typename> class Layer>
struct Sublayerof;
}