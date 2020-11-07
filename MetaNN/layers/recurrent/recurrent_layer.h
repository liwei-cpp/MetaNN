#pragma once
#include <MetaNN/facilities/cont_metafuns/_.h>
namespace MetaNN
{
    struct KernelSublayer;

    template <typename TPort> struct Previous;

    namespace NSRecurrentLayer
    {
        template <typename TKey>
        constexpr bool IsPreviousPort = false;

        template <typename TKey>
        constexpr bool IsPreviousPort<Previous<TKey>> = true;    
    }

    template <typename TPortName, size_t ID>
    struct SeqID : public Helper::KVBinder<TPortName, Helper::Int_<ID>>
    {
        static_assert(!NSRecurrentLayer::IsPreviousPort<TPortName>,
                      "Previous port should not correspond to sequence tensor");
    };

    namespace NSRecurrentLayer
    {
        template <typename T>
        constexpr static bool IsSeqID = false;

        template <typename TPortName, size_t ID>
        constexpr static bool IsSeqID<SeqID<TPortName, ID>> = true;
    }

    struct RnnPolicy
    {
        using MajorClass = RnnPolicy;
        struct SeqIdContTypeCate;
        
        struct UseBpttValueCate;
        constexpr static bool UseBptt = true;
    };
    
#include <MetaNN/policies/policy_macro_begin.h>
    ValuePolicyObj(PEnableBptt,  RnnPolicy, UseBptt, true);
    ValuePolicyObj(PDisableBptt,  RnnPolicy, UseBptt, false);
#include <MetaNN/policies/policy_macro_end.h>

    template <typename... TSeqIDs>
    struct PSeqIDsAre : virtual public RnnPolicy
    {
        using MinorClass = RnnPolicy::SeqIdContTypeCate;
        using SeqIdCont = PSeqIDsAre;
    };

    namespace NSRecurrentLayer
    {
        template <typename TInputSet, typename TOutputSet>
        struct CheckPortOverLap_;
        
        template <typename TInputSet, typename... TOutputPorts>
        struct CheckPortOverLap_<TInputSet, LayerPortSet<TOutputPorts...>>
        {
            constexpr static bool value = (Set::HasKey<TInputSet, Previous<TOutputPorts>> && ...);
        };

        template <typename TSeqIdCont, typename TInputPortSet>
        constexpr static bool SeqIdsValid = false;

        template <typename TInputPortSet, typename... TSeqIds>
        constexpr static bool SeqIdsValid<PSeqIDsAre<TSeqIds...>, TInputPortSet>
            = (Set::HasKey<TInputPortSet, typename TSeqIds::KeyType> && ...);

        constexpr size_t IDSwap(size_t seqID, size_t oriID)
        {
            return (oriID == 0) ? seqID :
                   ((oriID <= seqID) ? oriID - 1 : oriID);
        }

        template <size_t SeqID, typename TIndexes>
        struct GetPermuteInfo_;

        template <size_t SeqID, size_t... Indexes>
        struct GetPermuteInfo_<SeqID, std::index_sequence<Indexes...>>
        {
            using type = PDimArrayIs<IDSwap(SeqID, Indexes)...>;
        };

        template <typename TSeqInfo, bool IsPrevious, typename TValue>
        struct Wrapper2KernelInputMapHelper_
        {
            static_assert(!IsPrevious);
            constexpr static size_t ValueDim = DataCategory<TValue>::DimNum;
            constexpr static size_t ID = TSeqInfo::value;
            using PermutePolicy = PolicyContainer<typename GetPermuteInfo_<ID, std::make_index_sequence<ValueDim>>::type>;
            using PermuteType = decltype(Permute<PermutePolicy>(std::declval<TValue>()));
            using type = decltype(std::declval<PermuteType>()[0]);
        };
        
        template <typename TValue>
        struct Wrapper2KernelInputMapHelper_<void, false, TValue>
        {
            using type = TValue;
        };
        
        template <typename TValue>
        struct Wrapper2KernelInputMapHelper_<void, true, TValue>
        {
            using type = RemConstRef<decltype(MakeDynamic(std::declval<TValue>()))>;
        };        

        template <typename TInputMap, typename TSeqIdCont>
        struct Wrapper2KernelInputMap_;
        
        template <typename... TKeys, typename... TValues, typename TSeqIdCont>
        struct Wrapper2KernelInputMap_<LayerInMap<LayerKV<TKeys, TValues>...>, TSeqIdCont>
        {
            using type = 
                LayerInMap<LayerKV<TKeys, 
                                   typename Wrapper2KernelInputMapHelper_<Map::Find<TSeqIdCont, TKeys>, IsPreviousPort<TKeys>, TValues>::type>...>;
        };
        
        template <typename TInputMap, typename TSeqIDs>
        struct CalKernelInputMap_;
        
        template <typename TSeqIDs>
        struct CalKernelInputMap_<NullParameter, TSeqIDs>
        {
            using type = NullParameter;
        };
        
        template <typename... TKeys, typename... TValues, typename TSeqIDs>
        struct CalKernelInputMap_<LayerInMap<LayerKV<TKeys, TValues>...>, TSeqIDs>
        {
            static_assert((IsPreviousPort<TKeys> || ...), "No Previous port in the input port set.");
            using type = typename Wrapper2KernelInputMap_<LayerInMap<LayerKV<TKeys, TValues>...>,
                                                          TSeqIDs>::type;
        };
        
        template <typename TMap, typename TInputPortset>
        constexpr static bool InputMapPortsetMatch = false;
        
        template <typename... TKeys, typename... TValues, typename TInputPortset>
        constexpr static bool InputMapPortsetMatch<LayerInMap<LayerKV<TKeys, TValues>...>, TInputPortset>
            = Set::IsEqual<LayerInMap<TKeys...>, TInputPortset>;

        template <typename TInputMap, typename TPolicies>
        struct KernelGenerator_
        {
            using WrapperPolicy = PlainPolicy<TPolicies>;

            template <typename UInput, typename UPolicies>
            using Kernel = typename PolicySelect<LayerStructurePolicy, WrapperPolicy>::template ActFunc<UInput, UPolicies>;
            static_assert(!LayerStructurePolicy::template IsDummyActFun<Kernel>, "Use PActFuncIs<...> to set kernel sublayer.");

            using KernelPolicy = SubPolicyPicker<TPolicies, KernelSublayer>;
            constexpr static bool IsUpdate = PolicySelect<GradPolicy, KernelPolicy>::IsUpdate;
            constexpr static bool UseBptt = PolicySelect<RnnPolicy, KernelPolicy>::UseBptt;
            constexpr static bool UpdateFeedbackOutput =
                PolicySelect<GradPolicy, WrapperPolicy>::IsFeedbackOutput || (IsUpdate && UseBptt);

            using AmendKernelPolicy = typename std::conditional_t<UpdateFeedbackOutput,
                                                                  ChangePolicy_<PFeedbackOutput, KernelPolicy>,
                                                                  Identity_<KernelPolicy>>::type;

            using SeqIdCont = typename PolicySelect<RnnPolicy, WrapperPolicy>::SeqIdCont;
            using KernelInputMap = typename CalKernelInputMap_<TInputMap, SeqIdCont>::type;
            using KernelType = Kernel<KernelInputMap, AmendKernelPolicy>;
            using InputPortSet = typename KernelType::InputPortSet;
            using OutputPortSet = typename KernelType::OutputPortSet;
            
            using InputMap = typename std::conditional_t<std::is_same_v<KernelInputMap, NullParameter>,
                                                         EmptyLayerInMap_<InputPortSet>,
                                                         Identity_<TInputMap>>::type;
            static_assert(InputMapPortsetMatch<InputMap, InputPortSet>, "Invalid input port set.");
            static_assert(CheckPortOverLap_<InputPortSet, OutputPortSet>::value);
            static_assert(SeqIdsValid<SeqIdCont, InputPortSet>,
                          "Some sequence id not correcponds to any RNN port.");            
            static_assert(Sequential::Size<SeqIdCont> != 0,
                          "Trivial recurrent layer is not allowed.");

        };

        template <typename TKey>
        struct PreviousToPrimePort_;
        
        template <typename TKey>
        struct PreviousToPrimePort_<Previous<TKey>>
        {
            using type = TKey;
        };

        template <bool bFeedbackOutput, typename TInputMap>
        struct ShapeDictHelper
        {
            static_assert(!bFeedbackOutput);
            using type = NullParameter;
        };
        
        template <typename... TKeys, typename... TValues>
        struct ShapeDictHelper<true, LayerInMap<LayerKV<TKeys, TValues>...>>
        {
            using shapeDictType = typename VarTypeDict<TKeys...>::template Values<RemConstRef<decltype(std::declval<TValues>().Shape())>...>;
            using type = std::stack<shapeDictType>;
            
            template <typename Key, typename TIn, typename Cont>
            static void PickShapeInfoHelper(const TIn& p_in, Cont&& p_cont)
            {
                auto curShape = p_in.template Get<Key>().Shape();
                static_assert(std::is_same_v<decltype(curShape), typename shapeDictType::template ValueType<Key>>);
                p_cont.template Update<Key>(std::move(curShape));
            }
            
            template <typename TIn>
            static void PickShapeInfo(type& shapeStack, const TIn& p_in)
            {
                shapeDictType res;
                (PickShapeInfoHelper<TKeys>(p_in, res), ...);
                shapeStack.push(res);
            }
            
            template <typename TKeysCont, typename TSeqIdCont, typename TShapeCont, typename TCont>
            static auto CollapseHelper(const TShapeCont& p_shape, TCont&& p_cont)
            {
                if constexpr (Sequential::Size<TKeysCont> == 0)
                    return std::forward<TCont>(p_cont);
                else
                {
                    using CurType = Sequential::Head<TKeysCont>;
                    const auto& oriValue = p_cont.template Get<CurType>();
                    if constexpr (Map::HasKey<TSeqIdCont, CurType>)
                    {
                        constexpr size_t seqID = Map::Find<TSeqIdCont, CurType>::value;
                        constexpr static size_t ValueDim = DataCategory<decltype(oriValue)>::DimNum;
                        using PermutePolicy = typename GetPermuteInfo_<seqID, std::make_index_sequence<ValueDim>>::type;
                        auto newValue = PermuteInv<PolicyContainer<PermutePolicy>>(oriValue);
                        assert(newValue.Shape() == p_shape.template Get<CurType>());
                        auto newCont = std::forward<TCont>(p_cont).template Set<CurType>(newValue);
                        return CollapseHelper<Sequential::Tail<TKeysCont>, TSeqIdCont>(p_shape, std::move(newCont));
                    }
                    else if constexpr (!IsPreviousPort<CurType>)
                    {
                        auto newValue = ReduceSum<PolicyContainer<PModifyDimNumIs<1>>>(oriValue);
                        assert(newValue.Shape() == p_shape.template Get<CurType>());
                        auto newCont = std::forward<TCont>(p_cont).template Set<CurType>(newValue);
                        return CollapseHelper<Sequential::Tail<TKeysCont>, TSeqIdCont>(p_shape, std::move(newCont));
                    }
                    else
                    {
                        return CollapseHelper<Sequential::Tail<TKeysCont>, TSeqIdCont>(p_shape, std::forward<TCont>(p_cont));
                    }
                }
            }

            template <typename TSeqIdCont, typename TRes>
            static auto Collapse(type& shapeStack, TRes&& p_res)
            {
                assert(!shapeStack.empty());
                
                auto currentShapeDict = shapeStack.top();
                shapeStack.pop();
                return CollapseHelper<VarTypeDict<TKeys...>, TSeqIdCont>(currentShapeDict, std::forward<TRes>(p_res));
            }
        };
        
        template <typename TKeyCont, typename TSeqID, typename TInput, typename TOutput>
        auto PermuteBySeqID(const TInput& p_input, TOutput&& p_output)
        {
            if constexpr (Sequential::Size<TKeyCont> == 0)
                return std::forward<TOutput>(p_output);
            else
            {
                using CurType = Sequential::Head<TKeyCont>;
                auto curValue = p_input.template Get<CurType>();
                if constexpr (Map::HasKey<TSeqID, CurType>)
                {
                    constexpr size_t curSeqID = Map::Find<TSeqID, CurType>::value;
                    constexpr static size_t ValueDim = DataCategory<decltype(curValue)>::DimNum;
                    using PermutePolicy = typename GetPermuteInfo_<curSeqID, std::make_index_sequence<ValueDim>>::type;
                    auto newValue = Permute<PolicyContainer<PermutePolicy>>(std::move(curValue));
                    auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(newValue);
                    return PermuteBySeqID<Sequential::Tail<TKeyCont>, TSeqID>(p_input, std::move(newOutput));
                }
                else
                {
                    auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(curValue);
                    return PermuteBySeqID<Sequential::Tail<TKeyCont>, TSeqID>(p_input, std::move(newOutput));
                }
            }
        }
        
        template <typename TKeyCont, typename TSeqID, typename TIn>
        size_t GetSeqNum(const TIn& p_in, size_t prev = 0)
        {
            if constexpr (Sequential::Size<TKeyCont> == 0)
                return prev;
            else
            {
                using CurType = Sequential::Head<TKeyCont>;
                if constexpr(Map::HasKey<TSeqID, CurType>)
                {
                    size_t seqValue = p_in.template Get<CurType>().Shape()[0];
                    if (seqValue == 0)
                        throw std::runtime_error("Empty sequence as input.");
                    if (prev == 0) prev = seqValue;
                    else if ((prev != seqValue) && (seqValue != 0))
                    {
                        throw std::runtime_error("Sequence number mismatch.");
                    }
                }
                return GetSeqNum<Sequential::Tail<TKeyCont>, TSeqID>(p_in, prev);
            }
        }
        
        template <typename TKeyCont, typename TSeqID, typename TInput, typename TOutput>
        auto Split0(const TInput& p_input, TOutput&& p_output)
        {
            if constexpr (Sequential::Size<TKeyCont> == 0)
                return std::forward<TOutput>(p_output);
            else
            {
                using CurType = Sequential::Head<TKeyCont>;
                auto curValue = p_input.template Get<CurType>();
                if constexpr (Map::HasKey<TSeqID, CurType>)
                {
                    auto inputValue = curValue[0];
                    auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(inputValue);
                    return Split0<Sequential::Tail<TKeyCont>, TSeqID>(p_input, std::move(newOutput));
                }
                else if constexpr (IsPreviousPort<CurType>)
                {
                    auto inputValue = MakeDynamic(std::move(curValue));
                    auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(inputValue);
                    return Split0<Sequential::Tail<TKeyCont>, TSeqID>(p_input, std::move(newOutput));
                }
                else
                {
                    auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(curValue);
                    return Split0<Sequential::Tail<TKeyCont>, TSeqID>(p_input, std::move(newOutput));
                }
            }
        }
        
        template <typename TKeyCont, typename TSeqID, typename TInput, typename TPrevious, typename TOutput>
        auto SplitN(const TInput& p_input, TOutput&& p_output, const TPrevious& p_previous, size_t id)
        {
            assert(id != 0);

            if constexpr (Sequential::Size<TKeyCont> == 0)
                return std::forward<TOutput>(p_output);
            else
            {
                using CurType = Sequential::Head<TKeyCont>;
                if constexpr (IsPreviousPort<CurType>)
                {
                    using PrimeType = typename PreviousToPrimePort_<CurType>::type;
                    auto curValue = p_previous.template Get<PrimeType>();
                    auto inputValue = MakeDynamic(std::move(curValue));
                    auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(inputValue);
                    return SplitN<Sequential::Tail<TKeyCont>, TSeqID>(p_input, std::move(newOutput), p_previous, id);
                }
                else
                {
                    auto curValue = p_input.template Get<CurType>();
                    if constexpr (Map::HasKey<TSeqID, CurType>)
                    {
                        auto inputValue = curValue[id];
                        auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(inputValue);
                        return SplitN<Sequential::Tail<TKeyCont>, TSeqID>(p_input, std::move(newOutput), p_previous, id);
                    }
                    else
                    {
                        auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(curValue);
                        return SplitN<Sequential::Tail<TKeyCont>, TSeqID>(p_input, std::move(newOutput), p_previous, id);
                    }
                }
            }
        }
        
        template <typename TKeyCont, typename TInput, typename TOutput>
        auto InitOutputCont(TInput&& p_input, TOutput&& p_output)
        {
            if constexpr (Sequential::Size<TKeyCont> == 0)
                return std::forward<TOutput>(p_output);
            else
            {
                using CurType = Sequential::Head<TKeyCont>;
                using CurValueType = typename RemConstRef<TInput>::template ValueType<CurType>;
                ScalableTensor<CurValueType> aimValue;
                aimValue.PushBack(std::forward<TInput>(p_input).template Get<CurType>());
                auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(aimValue);
                return InitOutputCont<Sequential::Tail<TKeyCont>>(std::forward<TInput>(p_input), std::move(newOutput));
            }
        }
        
        template <typename TKeyCont, typename TInput, typename TOutput>
        void FillOutputCont(TInput&& p_input, TOutput&& p_output)
        {
            if constexpr (Sequential::Size<TKeyCont> == 0)
                return;
            else
            {
                using CurType = Sequential::Head<TKeyCont>;
                p_output.template Get<CurType>().PushBack(std::forward<TInput>(p_input).template Get<CurType>());
                return FillOutputCont<Sequential::Tail<TKeyCont>>(std::forward<TInput>(p_input), std::forward<TOutput>(p_output));
            }
        }

        template <typename TKeyCont, typename TIn>
        size_t GetGradSeqNum(const TIn& p_in, size_t prev = 0)
        {
            if constexpr (Sequential::Size<TKeyCont> == 0)
                return prev;
            else
            {
                using CurType = Sequential::Head<TKeyCont>;

                size_t seqValue = p_in.template Get<CurType>().Shape()[0];
                if (seqValue == 0)
                    throw std::runtime_error("Empty sequence as input.");
                if (prev == 0) prev = seqValue;
                else if ((prev != seqValue) && (seqValue != 0))
                {
                    throw std::runtime_error("Sequence number mismatch.");
                }

                return GetGradSeqNum<Sequential::Tail<TKeyCont>>(p_in, prev);
            }
        }
        
        template <typename TKeyCont, typename TInput, typename TOutput>
        auto GradSplit0(const TInput& p_input, TOutput&& p_output)
        {
            if constexpr (Sequential::Size<TKeyCont> == 0)
                return std::forward<TOutput>(p_output);
            else
            {
                using CurType = Sequential::Head<TKeyCont>;
                auto curValue = p_input.template Get<CurType>();

                auto inputValue = curValue[curValue.Shape()[0] - 1];
                auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(MakeDynamic(std::move(inputValue)));
                return GradSplit0<Sequential::Tail<TKeyCont>>(p_input, std::move(newOutput));
            }
        }
        
        template <typename TKeyCont, bool UseBptt, typename TInput, typename TPrevious, typename TOutput>
        auto GradSplitN(const TInput& p_input, const TPrevious& p_previous, TOutput&& p_output, size_t id)
        {
            if constexpr (Sequential::Size<TKeyCont> == 0)
                return std::forward<TOutput>(p_output);
            else
            {
                using CurType = Sequential::Head<TKeyCont>;
                auto curValue = p_input.template Get<CurType>();

                auto inputValue = curValue[id];
                if constexpr (!UseBptt)
                {
                    auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(MakeDynamic(std::move(inputValue)));
                    return GradSplitN<Sequential::Tail<TKeyCont>, UseBptt>(p_input, p_previous, std::move(newOutput), id);
                }
                else
                {
                    auto amendInput = p_previous.template Get<Previous<CurType>>() + inputValue;
                    auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(MakeDynamic(std::move(amendInput)));
                    return GradSplitN<Sequential::Tail<TKeyCont>, UseBptt>(p_input, p_previous, std::move(newOutput), id);
                }
            }
        }
        
        template <typename TKeyCont, typename TOutput>
        auto ReverseOutputCont(TOutput&& p_output)
        {
            if constexpr (Sequential::Size<TKeyCont> == 0)
                return;
            else
            {
                using CurType = Sequential::Head<TKeyCont>;
                if constexpr (!IsPreviousPort<CurType>)
                    p_output.template Get<CurType>().Reverse();
                return ReverseOutputCont<Sequential::Tail<TKeyCont>>(std::forward<TOutput>(p_output));
            }
        }
        
        template <typename TKeyCont, typename TInput, typename TOutput>
        void FillNormalGradOutput(TInput&& p_input, TOutput&& p_output)
        {
            if constexpr (Sequential::Size<TKeyCont> == 0)
                return;
            else
            {
                using CurType = Sequential::Head<TKeyCont>;
                if constexpr (!IsPreviousPort<CurType>)
                {
                    p_output.template Get<CurType>().PushBack(std::forward<TInput>(p_input).template Get<CurType>());
                }
                return FillNormalGradOutput<Sequential::Tail<TKeyCont>>(std::forward<TInput>(p_input), std::forward<TOutput>(p_output));
            }
        }

        template <typename TKeyCont, typename TInput, typename TOutput>
        auto FillPrevGradOutput(TInput&& p_input, TOutput&& p_output)
        {
            if constexpr (Sequential::Size<TKeyCont> == 0)
                return std::forward<TOutput>(p_output);
            else
            {
                using CurType = Sequential::Head<TKeyCont>;
                if constexpr (IsPreviousPort<CurType>)
                {
                    auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(std::forward<TInput>(p_input).template Get<CurType>());
                    return FillPrevGradOutput<Sequential::Tail<TKeyCont>>(std::forward<TInput>(p_input), std::move(newOutput));
                }
                else
                {
                    return FillPrevGradOutput<Sequential::Tail<TKeyCont>>(std::forward<TInput>(p_input),
                                                                          std::forward<TOutput>(p_output));
                }
            }
        }
    }
    
    template <typename TInputs, typename TPolicies>
    class RecurrentLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using CurrentPolicy = PlainPolicy<TPolicies>;
        using KernelGen = NSRecurrentLayer::KernelGenerator_<TInputs, TPolicies>;

        using KernelType = typename KernelGen::KernelType;
        constexpr static bool UseBptt = KernelGen::UseBptt;
    public:
        static constexpr bool IsFeedbackOutput = PolicySelect<GradPolicy, CurrentPolicy>::IsFeedbackOutput;
        static constexpr bool IsUpdate = KernelType::IsUpdate;

        using InputPortSet = typename KernelGen::InputPortSet;
        using OutputPortSet = typename KernelGen::OutputPortSet;
        using InputMap = typename KernelGen::InputMap;

    private:
        using SeqIdCont = typename PolicySelect<RnnPolicy, CurrentPolicy>::SeqIdCont;
        using TShapeDickHelper = typename NSRecurrentLayer::ShapeDictHelper<IsFeedbackOutput, InputMap>;

    public:
        template <typename... TParams>
        RecurrentLayer(const std::string& p_name, TParams&&... kernelParams)
            : m_name(p_name)
            , m_kernel(p_name + "/kernel", std::forward<TParams>(kernelParams)...)
        {}

        template <typename TInitializer, typename TBuffer>
        void Init(TInitializer& initializer, TBuffer& loadBuffer)
        {
            LayerInit(m_kernel, initializer, loadBuffer);
        }
        
        template <typename TSave>
        void SaveWeights(TSave& saver) const
        {
            LayerSaveWeights(m_kernel, saver);
        }
        
        template <typename TGradCollector>
        void GradCollect(TGradCollector& col)
        {
            LayerGradCollect(m_kernel, col);
        }

        void NeutralInvariant() const
        {
            if constexpr (IsFeedbackOutput)
            {
                if (!m_inputShapeStack.empty())
                {
                    throw std::runtime_error("[RecurrentLayer] NeutralInvariant fails: ShapeStack is not empty.");
                }
            }
            LayerNeutralInvariant(m_kernel);
        }
        
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            using TInputKeys = typename RemConstRef<TIn>::Keys;
            if constexpr (IsFeedbackOutput)
            {
                TShapeDickHelper::PickShapeInfo(m_inputShapeStack, p_in);
            }

            auto permuteRes = NSRecurrentLayer::PermuteBySeqID<TInputKeys, SeqIdCont>(std::forward<TIn>(p_in), TInputKeys::Create());
            const size_t seqNum = NSRecurrentLayer::GetSeqNum<TInputKeys, SeqIdCont>(permuteRes);
            if (seqNum == 0)
            {
                throw std::runtime_error("Empty sequence as input.");
            }

            auto firstInputCont = NSRecurrentLayer::Split0<TInputKeys, SeqIdCont>(permuteRes, TInputKeys::Create());
            auto previousOutput = m_kernel.FeedForward(std::move(firstInputCont));
            using OutputKeys = typename decltype(previousOutput)::Keys;
            auto outputCont = NSRecurrentLayer::InitOutputCont<OutputKeys>(previousOutput, OutputKeys::Create());

            for (size_t i = 1; i < seqNum; ++i)
            {
                auto curInputCont = NSRecurrentLayer::SplitN<TInputKeys, SeqIdCont>(permuteRes, TInputKeys::Create(), previousOutput, i);
                previousOutput = m_kernel.FeedForward(std::move(curInputCont));
                NSRecurrentLayer::FillOutputCont<OutputKeys>(previousOutput, outputCont);
            }
            return outputCont;
        }

        template <typename TIn>
        auto FeedBackward(TIn&& p_in)
        {
            if constexpr (UseBptt)
                static_assert(KernelType::IsFeedbackOutput);

            using TInputKeys = typename RemConstRef<TIn>::Keys;
            if constexpr (!IsFeedbackOutput && !IsUpdate)
            {
                static_assert(!KernelType::IsFeedbackOutput);
                return LayerInputCont<RecurrentLayer>();
            }
            else if constexpr (!IsFeedbackOutput)
            {
                const size_t seqNum = NSRecurrentLayer::GetGradSeqNum<TInputKeys>(p_in);
                if (seqNum == 0)
                {
                    throw std::runtime_error("Empty sequence as grad input.");
                }
                
                auto curInputCont = NSRecurrentLayer::GradSplit0<TInputKeys>(p_in, TInputKeys::Create());
                auto curOutputCont = m_kernel.FeedBackward(std::move(curInputCont));

                for (size_t i = 2; i <= seqNum; ++i)
                {
                    auto curInputCont = NSRecurrentLayer::GradSplitN<TInputKeys, UseBptt>(p_in, curOutputCont, TInputKeys::Create(), seqNum - i);
                    curOutputCont = m_kernel.FeedBackward(std::move(curInputCont));
                }
                
                return LayerInputCont<RecurrentLayer>();
            }
            else
            {
                const size_t seqNum = NSRecurrentLayer::GetGradSeqNum<TInputKeys>(p_in);
                if (seqNum == 0)
                {
                    throw std::runtime_error("Empty sequence as grad input.");
                }
                
                auto firstInputGrad = NSRecurrentLayer::GradSplit0<TInputKeys>(p_in, TInputKeys::Create());
                auto curOutputCont = m_kernel.FeedBackward(std::move(firstInputGrad));
                using OutputGradKeys = typename decltype(curOutputCont)::Keys;
                auto outputGrad = NSRecurrentLayer::InitOutputCont<OutputGradKeys>(curOutputCont, OutputGradKeys::Create());
                for (size_t i = 2; i <= seqNum; ++i)
                {
                    auto curInputCont = NSRecurrentLayer::GradSplitN<TInputKeys, UseBptt>(p_in, curOutputCont, TInputKeys::Create(), seqNum - i);
                    curOutputCont = m_kernel.FeedBackward(std::move(curInputCont));
                    NSRecurrentLayer::FillNormalGradOutput<OutputGradKeys>(curOutputCont, outputGrad);
                }
                
                auto filledPrevGrad = NSRecurrentLayer::FillPrevGradOutput<OutputGradKeys>(curOutputCont, std::move(outputGrad));

                NSRecurrentLayer::ReverseOutputCont<OutputGradKeys>(filledPrevGrad);
                return TShapeDickHelper::template Collapse<SeqIdCont>(m_inputShapeStack, std::move(filledPrevGrad));
            }
        }
    private:
        std::string m_name;
        KernelType m_kernel;
        
        typename TShapeDickHelper::type m_inputShapeStack;
    };
}