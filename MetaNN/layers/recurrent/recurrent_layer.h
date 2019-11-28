#pragma once

namespace MetaNN
{
    template <typename TPort> struct Previous;
    
    namespace NSRecurrentLayer
    {
        template <typename T>
        constexpr bool IsBatchCategory = IsBatchCategoryTag<typename T::CategoryTag> || IsBatchSequenceCategoryTag<typename T::CategoryTag>;
        
        template <typename T>
        constexpr bool IsSequenceCategory = IsSequenceCategoryTag<typename T::CategoryTag>;

        template <typename TInputSet, typename TOutputSet>
        struct CheckPortOverLap_;
        
        template <typename TInputSet, typename... TOutputPorts>
        struct CheckPortOverLap_<TInputSet, LayerPortSet<TOutputPorts...>>
        {
            constexpr static bool value = (Set::HasKey<TInputSet, Previous<TOutputPorts>> && ...);
        };
        
        template <typename TInputMap, typename TPolicies>
        struct KernelGenerator_;
        
        template <typename TPolicies>
        struct KernelGenerator_<NullParameter, TPolicies>
        {
            using WrapperPolicy = PlainPolicy<TPolicies>;
            
            template <typename UInput, typename UPolicies>
            using Kernel = typename PolicySelect<LayerStructurePolicy, WrapperPolicy>::template ActFunc<UInput, UPolicies>;
            static_assert(!LayerStructurePolicy::template IsDummyActFun<Kernel>, "Use PActFuncIs<...> to set kernel sublayer.");

            using KernelPolicy = SubPolicyPicker<TPolicies, KernelSublayer>;

            constexpr static bool IsUpdate = PolicySelect<GradPolicy, KernelPolicy>::IsUpdate;
            constexpr static bool UseBptt = PolicySelect<RecurrentLayerPolicy, KernelPolicy>::UseBptt;
            constexpr static bool UpdateFeedbackOutput = PolicySelect<GradPolicy, WrapperPolicy>::IsFeedbackOutput ||
                                                         (IsUpdate && UseBptt);
            using AmendKernelPolicy = typename std::conditional_t<UpdateFeedbackOutput,
                                                                  ChangePolicy_<PFeedbackOutput, KernelPolicy>,
                                                                  Identity_<KernelPolicy>>::type;

            using KernelType = Kernel<NullParameter, AmendKernelPolicy>;
            
            using InputPortSet = typename KernelType::InputPortSet;
            using OutputPortSet = typename KernelType::OutputPortSet;
            static_assert(CheckPortOverLap_<InputPortSet, OutputPortSet>::value);
            
            using InputMap = EmptyLayerIOMap_<InputPortSet>;
            
            constexpr static bool IsTrival = false;
        };
        
        template <typename TKey>
        constexpr bool IsPreviousPort = false;
        
        template <typename TKey>
        constexpr bool IsPreviousPort<Previous<TKey>> = true;
        
        template <typename TKey>
        struct PreviousToPrimePort_;
        
        template <typename TKey>
        struct PreviousToPrimePort_<Previous<TKey>>
        {
            using type = TKey;
        };
        
        template <typename TValue, bool IsSequence, bool IsPrevious>
        struct Wrapper2KernelInputMapHelper_
        {
            static_assert(!(IsSequence && IsPrevious), "Previous<> is sequence.");
            using type = TValue;
        };
        
        template <typename TValue>
        struct Wrapper2KernelInputMapHelper_<TValue, true, false>
        {
            using type = decltype(std::declval<TValue>().operator[](0));
        };
        
        template <typename TValue>
        struct Wrapper2KernelInputMapHelper_<TValue, false, true>
        {
            using type = decltype(MakeDynamic(std::declval<TValue>()));
        };
        
        template <typename TInputMap>
        struct Wrapper2KernelInputMap_;
        
        template <typename... TKeys, typename... TValues>
        struct Wrapper2KernelInputMap_<LayerIOMap<LayerKV<TKeys, TValues>...>>
        {
            using type = 
                LayerIOMap<LayerKV<TKeys, 
                           typename Wrapper2KernelInputMapHelper_<TValues, IsSequenceCategory<TValues>, IsPreviousPort<TKeys>>::type>...>;
        };
        
        template <typename TKey, typename TValue>
        constexpr bool PreviousSeqCheck = !(IsPreviousPort<TKey> && IsSequenceCategoryTag<typename TValue::CategoryTag>);
        
        template <typename... TKeys, typename... TValues, typename TPolicies>
        struct KernelGenerator_<LayerIOMap<LayerKV<TKeys, TValues>...>, TPolicies>
        {
            static_assert((!IsBatchCategory<TValues> && ...), "No batch input is allowed in RNN layer.");
            static_assert((IsPreviousPort<TKeys> || ...), "No Previous port in the input port set.");
            static_assert((PreviousSeqCheck<TKeys, TValues> && ...),
                          "Previous ports should not be sequencal.");

            constexpr static bool IsTrival = !(IsSequenceCategory<TValues> || ...);
            
            using WrapperPolicy = PlainPolicy<TPolicies>;
            
            template <typename UInput, typename UPolicies>
            using Kernel = typename PolicySelect<LayerStructurePolicy, WrapperPolicy>::template ActFunc<UInput, UPolicies>;
            static_assert(!LayerStructurePolicy::template IsDummyActFun<Kernel>, "Use PActFuncIs<...> to set kernel sublayer.");

            using KernelPolicy = SubPolicyPicker<TPolicies, KernelSublayer>;
            
            constexpr static bool IsUpdate = PolicySelect<GradPolicy, KernelPolicy>::IsUpdate;
            constexpr static bool UseBptt = PolicySelect<RecurrentLayerPolicy, KernelPolicy>::UseBptt;
            constexpr static bool UpdateFeedbackOutput =
                PolicySelect<GradPolicy, WrapperPolicy>::IsFeedbackOutput || (!IsTrival && IsUpdate && UseBptt);
                
            using AmendKernelPolicy = typename std::conditional_t<UpdateFeedbackOutput,
                                                                  ChangePolicy_<PFeedbackOutput, KernelPolicy>,
                                                                  Identity_<KernelPolicy>>::type;

            using KernelInputMap = typename Wrapper2KernelInputMap_<LayerIOMap<LayerKV<TKeys, TValues>...>>::type;
            using KernelType = Kernel<KernelInputMap, AmendKernelPolicy>;

            using InputPortSet = typename KernelType::InputPortSet;
            using OutputPortSet = typename KernelType::OutputPortSet;
            static_assert(CheckPortOverLap_<InputPortSet, OutputPortSet>::value);
            static_assert(Set::IsEqual<LayerPortSet<TKeys...>, InputPortSet>, "Invalid input port set.");
            
            using InputMap = LayerIOMap<LayerKV<TKeys, TValues>...>;
        };
        
        template <typename TKeys, typename TInputCont>
        struct IsNonBatch_;
        
        template <typename TInputCont, typename... TKeys>
        struct IsNonBatch_<VarTypeDict<TKeys...>, TInputCont>
        {
            static constexpr bool value = 
                !(IsBatchCategory<typename TInputCont::template ValueType<TKeys>> || ...);
        };
        
        template <typename TKeys, typename TInputCont>
        struct IsNonSeq_;
        
        template <typename TInputCont, typename... TKeys>
        struct IsNonSeq_<VarTypeDict<TKeys...>, TInputCont>
        {
            static constexpr bool value = 
                !(IsSequenceCategory<typename TInputCont::template ValueType<TKeys>> || ...);
        };
        
        template <typename TKeys, typename TInputCont>
        struct IsAllSeq_;
        
        template <typename TInputCont, typename... TKeys>
        struct IsAllSeq_<VarTypeDict<TKeys...>, TInputCont>
        {
            static constexpr bool value = (IsSequenceCategory<typename TInputCont::template ValueType<TKeys>> && ...);
        };
        
        template <bool bFeedbackOutput, typename TInputMap>
        struct ShapeDictHelper
        {
            static_assert(!bFeedbackOutput);
            using type = NullParameter;

            template <typename TIn>
            static void PickShapeInfo(type&, const TIn&) {}
            
            template <typename TRes>
            static auto Collapse(type&, TRes&& p_res)
            {
                return std::forward<TRes>(p_res);
            }
        };
        
        template <typename... TKeys, typename... TValues>
        struct ShapeDictHelper<true, LayerIOMap<LayerKV<TKeys, TValues>...>>
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
            
            template <typename TKeysCont, typename TShapeCont, typename TCont>
            static auto CollapseHelper(const TShapeCont& p_shape, TCont&& p_cont)
            {
                if constexpr (Sequential::Size<TKeysCont> == 0)
                    return std::forward<TCont>(p_cont);
                else
                {
                    using CurType = Sequential::Head<TKeysCont>;
                    const auto& oriValue = p_cont.template Get<CurType>();
                    if constexpr (!std::is_same_v<RemConstRef<decltype(oriValue.Shape())>,
                                                  typename TShapeCont::template ValueType<CurType>>)
                    {
                        auto newValue = MetaNN::Collapse(oriValue, p_shape.template Get<CurType>());
                        auto newCont = std::forward<TCont>(p_cont).template Set<CurType>(newValue);
                        return CollapseHelper<Sequential::Tail<TKeysCont>>(p_shape, std::move(newCont));
                    }
                    else
                        return CollapseHelper<Sequential::Tail<TKeysCont>>(p_shape, std::forward<TCont>(p_cont));
                }
            }

            template <typename TRes>
            static auto Collapse(type& shapeStack, TRes&& p_res)
            {
                assert(!shapeStack.empty());
                
                auto currentShapeDict = shapeStack.top();
                shapeStack.pop();
                return CollapseHelper<VarTypeDict<TKeys...>>(currentShapeDict, std::forward<TRes>(p_res));
            }
        };
        
        template <typename TKeyCont, typename TIn, size_t pos = 0>
        size_t GetSeqNum(const TIn& p_in, size_t prev = 0)
        {
            if constexpr (pos == Sequential::Size<TKeyCont>)
            {
                return prev;
            }
            else
            {
                using TCurKey = Sequential::At<TKeyCont, pos>;
                using TCurValue = typename TIn::template ValueType<TCurKey>;
                
                size_t seqValue = 0;
                if constexpr (IsSequenceCategory<TCurValue>)
                {
                    seqValue = p_in.template Get<TCurKey>().Shape().Length();
                    if (seqValue == 0)
                        throw std::runtime_error("Empty sequence as input.");
                }

                if (prev == 0) prev = seqValue;
                else if ((prev != seqValue) && (seqValue != 0))
                {
                    throw std::runtime_error("Sequence number mismatch.");
                }
                return GetSeqNum<TKeyCont, TIn, pos + 1>(p_in, prev);
            }
        }
        
        template <typename TKeyCont, typename TInput, typename TOutput>
        auto Split0(const TInput& p_input, TOutput&& p_output)
        {
            if constexpr (Sequential::Size<TKeyCont> == 0)
                return std::forward<TOutput>(p_output);
            else
            {
                using CurType = Sequential::Head<TKeyCont>;
                auto curValue = p_input.template Get<CurType>();
                if constexpr (IsSequenceCategory<decltype(curValue)>)
                {
                    auto inputValue = curValue[0];
                    auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(inputValue);
                    return Split0<Sequential::Tail<TKeyCont>>(p_input, std::move(newOutput));
                }
                else if constexpr (IsPreviousPort<CurType>)
                {
                    auto inputValue = MakeDynamic(std::move(curValue));
                    auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(inputValue);
                    return Split0<Sequential::Tail<TKeyCont>>(p_input, std::move(newOutput));
                }
                else
                {
                    auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(curValue);
                    return Split0<Sequential::Tail<TKeyCont>>(p_input, std::move(newOutput));
                }
            }
        }
        
        template <typename TKeyCont, typename TInput, typename TPrevious, typename TOutput>
        auto SplitN(const TInput& p_input, TOutput&& p_output, const TPrevious& p_previous, size_t id)
        {
            assert(id != 0);

            if constexpr (Sequential::Size<TKeyCont> == 0)
                return std::forward<TOutput>(p_output);
            else
            {
                using CurType = Sequential::Head<TKeyCont>;
                using CurValueType = typename RemConstRef<TInput>::template ValueType<CurType>;
                if constexpr (IsPreviousPort<CurType>)
                {
                    using PrimeType = typename PreviousToPrimePort_<CurType>::type;
                    auto curValue = p_previous.template Get<PrimeType>();
                    auto inputValue = MakeDynamic(std::move(curValue));
                    auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(inputValue);
                    return SplitN<Sequential::Tail<TKeyCont>>(p_input, std::move(newOutput), p_previous, id);
                }
                else
                {
                    auto curValue = p_input.template Get<CurType>();
                    if constexpr (IsSequenceCategory<CurValueType>)
                    {
                        auto inputValue = curValue[id];
                        auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(inputValue);
                        return SplitN<Sequential::Tail<TKeyCont>>(p_input, std::move(newOutput), p_previous, id);
                    }
                    else
                    {
                        auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(curValue);
                        return SplitN<Sequential::Tail<TKeyCont>>(p_input, std::move(newOutput), p_previous, id);
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
                DynamicSequence<CurValueType> aimValue;
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
        
        template <typename TKeyCont, typename TInput, typename TOutput>
        auto GradSplit(const TInput& p_input, TOutput&& p_output)
        {
            if constexpr (Sequential::Size<TKeyCont> == 0)
                return std::forward<TOutput>(p_output);
            else
            {
                using CurType = Sequential::Head<TKeyCont>;
                auto curValue = p_input.template Get<CurType>();
                static_assert(IsSequenceCategory<decltype(curValue)>);
                
                auto inputValue = curValue[curValue.Shape().Length() - 1];
                auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(MakeDynamic(std::move(inputValue)));
                return GradSplit<Sequential::Tail<TKeyCont>>(p_input, std::move(newOutput));
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
                static_assert(IsSequenceCategory<decltype(curValue)>);
                
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
        
        constexpr static bool IsTrivalLayer = KernelGen::IsTrival;
    private:
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
            using TOriInputCont = RemConstRef<TIn>;
            static_assert(NSRecurrentLayer::IsNonBatch_<typename TOriInputCont::Keys, TOriInputCont>::value);
            if constexpr (IsTrivalLayer)
            {
                static_assert(NSRecurrentLayer::IsNonSeq_<typename TOriInputCont::Keys, TOriInputCont>::value);
                return m_kernel.FeedForward(p_in);
            }
            else
            {
                const size_t seqNum = NSRecurrentLayer::GetSeqNum<typename TOriInputCont::Keys>(p_in);
                if (seqNum == 0)
                {
                    throw std::runtime_error("Empty sequence as input.");
                }
                
                TShapeDickHelper::PickShapeInfo(m_inputShapeStack, p_in);
                
                auto firstInputCont = NSRecurrentLayer::Split0<typename TOriInputCont::Keys>(p_in, TOriInputCont::Keys::Create());
                auto previousOutput = m_kernel.FeedForward(std::move(firstInputCont));
                using OutputKeys = typename decltype(previousOutput)::Keys;
                auto outputCont = NSRecurrentLayer::InitOutputCont<OutputKeys>(previousOutput, OutputKeys::Create());

                for (size_t i = 1; i < seqNum; ++i)
                {
                    auto curInputCont = NSRecurrentLayer::SplitN<typename TOriInputCont::Keys>(p_in, TOriInputCont::Keys::Create(), previousOutput, i);
                    previousOutput = m_kernel.FeedForward(std::move(curInputCont));
                    NSRecurrentLayer::FillOutputCont<OutputKeys>(previousOutput, outputCont);
                }
                return outputCont;
            }
        }

        template <typename TIn>
        auto FeedBackward(TIn&& p_in)
        {
            if constexpr (UseBptt)
                static_assert(KernelType::IsFeedbackOutput);

            using TOriInputCont = RemConstRef<TIn>;
            if constexpr (IsTrivalLayer)
            {
                static_assert(NSRecurrentLayer::IsNonBatch_<typename TOriInputCont::Keys, TOriInputCont>::value);
                return m_kernel.FeedBackward(p_in);
            }
            else if constexpr (!IsFeedbackOutput && !IsUpdate)
            {
                static_assert(!KernelType::IsFeedbackOutput);
                return LayerInputCont<RecurrentLayer>();
            }
            else if constexpr (!IsFeedbackOutput)
            {
                static_assert(NSRecurrentLayer::IsAllSeq_<typename TOriInputCont::Keys, TOriInputCont>::value, "All grads should be Sequencal.");
                const size_t seqNum = NSRecurrentLayer::GetSeqNum<typename TOriInputCont::Keys>(p_in);
                if (seqNum == 0)
                {
                    throw std::runtime_error("Empty sequence as grad input.");
                }
                
                auto curInputCont = NSRecurrentLayer::GradSplit<typename TOriInputCont::Keys>(p_in, TOriInputCont::Keys::Create());
                auto curOutputCont = m_kernel.FeedBackward(std::move(curInputCont));

                for (size_t i = 2; i <= seqNum; ++i)
                {
                    auto curInputCont = NSRecurrentLayer::GradSplitN<typename TOriInputCont::Keys, UseBptt>(p_in, curOutputCont, TOriInputCont::Keys::Create(), seqNum - i);
                    curOutputCont = m_kernel.FeedBackward(std::move(curInputCont));
                }
                
                return LayerInputCont<RecurrentLayer>();
            }
            else
            {
                static_assert(NSRecurrentLayer::IsAllSeq_<typename TOriInputCont::Keys, TOriInputCont>::value, "All grads should be Sequencal.");
                const size_t seqNum = NSRecurrentLayer::GetSeqNum<typename TOriInputCont::Keys>(p_in);
                if (seqNum == 0)
                {
                    throw std::runtime_error("Empty sequence as grad input.");
                }
                
                auto firstInputGrad = NSRecurrentLayer::GradSplit<typename TOriInputCont::Keys>(p_in, TOriInputCont::Keys::Create());
                auto curOutputCont = m_kernel.FeedBackward(std::move(firstInputGrad));
                using OutputGradKeys = typename decltype(curOutputCont)::Keys;
                auto outputGrad = NSRecurrentLayer::InitOutputCont<OutputGradKeys>(curOutputCont, OutputGradKeys::Create());
                for (size_t i = 2; i <= seqNum; ++i)
                {
                    auto curInputCont = NSRecurrentLayer::GradSplitN<typename TOriInputCont::Keys, UseBptt>(p_in, curOutputCont, TOriInputCont::Keys::Create(), seqNum - i);
                    curOutputCont = m_kernel.FeedBackward(std::move(curInputCont));
                    NSRecurrentLayer::FillNormalGradOutput<OutputGradKeys>(curOutputCont, outputGrad);
                }
                
                auto filledPrevGrad = NSRecurrentLayer::FillPrevGradOutput<OutputGradKeys>(curOutputCont, std::move(outputGrad));

                NSRecurrentLayer::ReverseOutputCont<OutputGradKeys>(filledPrevGrad);
                return TShapeDickHelper::Collapse(m_inputShapeStack, std::move(filledPrevGrad));
            }
        }
    private:
        std::string m_name;
        KernelType m_kernel;
        
        typename TShapeDickHelper::type m_inputShapeStack;
    };
}