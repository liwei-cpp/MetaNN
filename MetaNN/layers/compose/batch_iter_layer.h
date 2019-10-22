#pragma once
#include <type_traits>

namespace MetaNN
{
    struct KernelSublayer;

    namespace NSBatchIterLayer
    {
        template <typename T>
        constexpr bool IsBatchCategory = IsBatchCategoryTag<typename T::CategoryTag> || IsBatchSequenceCategoryTag<typename T::CategoryTag>;

        template <typename T>
        struct RemoveBatchHelper_
        {
            using type = decltype(std::declval<T>().operator[](0));
        };
        
        template <typename TOriKV>
        struct RemoveBatchInfo_;
        
        template <typename TKey, typename TValue>
        struct RemoveBatchInfo_<LayerKV<TKey, TValue>>
        {
            using ValueType = typename std::conditional_t<IsBatchCategory<TValue>,
                                                          RemoveBatchHelper_<TValue>,
                                                          Identity_<TValue>>::type;
            using type = LayerKV<TKey, ValueType>;
        };
        
        template <typename TInputs, typename TPolicies>
        struct CtorKernelLayer_
        {
            using WrapperPolicy = PlainPolicy<TPolicies>;
            
            template <typename UInput, typename UPolicies>
            using Kernel = typename PolicySelect<LayerStructurePolicy, WrapperPolicy>::template ActFunc<UInput, UPolicies>;
            static_assert(!LayerStructurePolicy::template IsDummyActFun<Kernel>, "Use PActFuncIs<...> to set kernel sublayer.");

            using KernelInputMap = typename std::conditional_t<std::is_same_v<TInputs, NullParameter>,
                                                               Identity_<NullParameter>,
                                                               ContMetaFun::Sequential::Transform_<TInputs, RemoveBatchInfo_, LayerIOMap>>::type;

            using KernelPolicy = SubPolicyPicker<TPolicies, KernelSublayer>;
            using AmendKernelPolicy = typename std::conditional_t<PolicySelect<GradPolicy, WrapperPolicy>::IsFeedbackOutput,
                                                                  ChangePolicy_<PFeedbackOutput, KernelPolicy>,
                                                                  Identity_<KernelPolicy>>::type;

            using type = Kernel<KernelInputMap, AmendKernelPolicy>;
        };
        
        template <typename TInputs>
        struct IsIOMapNonBatch_;
        
        template <>
        struct IsIOMapNonBatch_<NullParameter>
        {
            constexpr static bool value = false;
        };
        
        template <typename... TKeys, typename... TValues>
        struct IsIOMapNonBatch_<LayerIOMap<LayerKV<TKeys, TValues>...>>
        {
            constexpr static bool value = (!IsBatchCategory<TValues> || ...);
        };
        
        template <typename TKeys, typename TInputCont>
        struct IsFBContNonBatch_;
        
        template <typename TInputCont, typename... TKeys>
        struct IsFBContNonBatch_<VarTypeDict<TKeys...>, TInputCont>
        {
            static constexpr bool value = 
                !(IsBatchCategory<typename TInputCont::template ValueType<TKeys>> || ...);
        };
        
        template <typename TKeys, typename TInputCont>
        struct IsFBContAllBatch_;
        
        template <typename TInputCont, typename... TKeys>
        struct IsFBContAllBatch_<VarTypeDict<TKeys...>, TInputCont>
        {
            static constexpr bool value = (IsBatchCategory<typename TInputCont::template ValueType<TKeys>> && ...);
        };
        
        template <typename TKeyCont, typename TIn, size_t pos = 0>
        size_t GetBatchNum(const TIn& p_in, size_t prev = 0)
        {
            if constexpr (pos == ArraySize<TKeyCont>)
            {
                return prev;
            }
            else
            {
                using TCurKey = ContMetaFun::Sequential::At<TKeyCont, pos>;
                using TCurValue = typename TIn::template ValueType<TCurKey>;
                
                size_t batchValue = 0;
                if constexpr (IsBatchCategory<TCurValue>)
                {
                    batchValue = p_in.template Get<TCurKey>().Shape().BatchNum();
                    if (batchValue == 0)
                        throw std::runtime_error("Empty batch value as input.");
                }

                if (prev == 0) prev = batchValue;
                else if (prev != batchValue)
                {
                    throw std::runtime_error("Batch number mismatch.");
                }
                return GetBatchNum<TKeyCont, TIn, pos + 1>(p_in, prev);
            }
        }
        
        template <typename TKeyCont, typename TInput, typename TOutput>
        auto Split(const TInput& p_input, TOutput&& p_output, size_t id)
        {
            if constexpr (ArraySize<TKeyCont> == 0)
                return std::forward<TOutput>(p_output);
            else
            {
                using CurType = SeqHead<TKeyCont>;
                auto curValue = p_input.template Get<CurType>();
                if constexpr (IsBatchCategory<decltype(curValue)>)
                {
                    auto inputValue = curValue[id];
                    auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(inputValue);
                    return Split<SeqTail<TKeyCont>>(p_input, std::move(newOutput), id);
                }
                else
                {
                    auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(curValue);
                    return Split<SeqTail<TKeyCont>>(p_input, std::move(newOutput), id);
                }
            }
        }
        
        template <typename TKeyCont, typename TInput, typename TOutput>
        auto InitOutputCont(TInput&& p_input, TOutput&& p_output)
        {
            if constexpr (ArraySize<TKeyCont> == 0)
                return std::forward<TOutput>(p_output);
            else
            {
                using CurType = SeqHead<TKeyCont>;
                using CurValueType = typename RemConstRef<TInput>::template ValueType<CurType>;
                DynamicBatch<CurValueType> aimValue;
                aimValue.PushBack(std::forward<TInput>(p_input).template Get<CurType>());
                auto newOutput = std::forward<TOutput>(p_output).template Set<CurType>(aimValue);
                return InitOutputCont<SeqTail<TKeyCont>>(std::forward<TInput>(p_input), std::move(newOutput));
            }
        }

        template <typename TKeyCont, typename TInput, typename TOutput>
        auto FillOutputCont(TInput&& p_input, TOutput&& p_output)
        {
            if constexpr (ArraySize<TKeyCont> == 0)
                return std::forward<TOutput>(p_output);
            else
            {
                using CurType = SeqHead<TKeyCont>;
                p_output.template Get<CurType>().PushBack(std::forward<TInput>(p_input).template Get<CurType>());
                return FillOutputCont<SeqTail<TKeyCont>>(std::forward<TInput>(p_input), std::forward<TOutput>(p_output));
            }
        }
        
        template <typename TKeyCont, typename TOutput>
        auto ReverseOutputCont(TOutput&& p_output)
        {
            if constexpr (ArraySize<TKeyCont> == 0)
                return std::forward<TOutput>(p_output);
            else
            {
                using CurType = SeqHead<TKeyCont>;
                p_output.template Get<CurType>().Reverse();
                return ReverseOutputCont<SeqTail<TKeyCont>>(std::forward<TOutput>(p_output));
            }
        }
        
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
                if constexpr (ArraySize<TKeysCont> == 0)
                    return std::forward<TCont>(p_cont);
                else
                {
                    using CurType = SeqHead<TKeysCont>;
                    const auto& oriValue = p_cont.template Get<CurType>();
                    auto newValue = MetaNN::Collapse(oriValue, p_shape.template Get<CurType>());
                    if constexpr (!std::is_same_v<RemConstRef<decltype(newValue)>, RemConstRef<decltype(oriValue)>>)
                    {
                        auto newCont = std::forward<TCont>(p_cont).template Set<CurType>(newValue);
                        return CollapseHelper<SeqTail<TKeysCont>>(p_shape, std::move(newCont));
                    }
                    else
                        return CollapseHelper<SeqTail<TKeysCont>>(p_shape, std::forward<TCont>(p_cont));
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
    }
    
    template <typename TInputs, typename TPolicies>
    class BatchIterLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using KernelType = typename NSBatchIterLayer::CtorKernelLayer_<TInputs, TPolicies>::type;

    public:
        static constexpr bool IsFeedbackOutput = KernelType::IsFeedbackOutput;
        static constexpr bool IsUpdate = KernelType::IsUpdate;

        using InputPortSet = typename KernelType::InputPortSet;
        using OutputPortSet = typename KernelType::OutputPortSet;
        using InputMap = typename std::conditional_t<std::is_same_v<TInputs, NullParameter>,
                                                     EmptyLayerIOMap_<InputPortSet>,
                                                     Identity_<TInputs>>::type;
        static_assert(CheckInputMapAvailable_<InputMap, InputPortSet>::value);

        static constexpr bool IsTrivalLayer = NSBatchIterLayer::IsIOMapNonBatch_<TInputs>::value;

    private:
        using TShapeDickHelper = typename NSBatchIterLayer::ShapeDictHelper<IsFeedbackOutput, InputMap>;

    public:
        template <typename... TParams>
        BatchIterLayer(const std::string& p_name, TParams&&... kernelParams)
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
            LayerNeutralInvariant(m_kernel);
            LayerTraits::CheckStackEmpty(m_inputShapeStack);
        }
    
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            using TOriInputCont = RemConstRef<TIn>;
            if constexpr (IsTrivalLayer)
            {
                static_assert(NSBatchIterLayer::IsFBContNonBatch_<typename TOriInputCont::Keys, TOriInputCont>::value);
                return m_kernel.FeedForward(p_in);
            }
            else
            {
                const size_t batchNum = NSBatchIterLayer::GetBatchNum<typename TOriInputCont::Keys>(p_in);
                if (batchNum == 0)
                {
                    throw std::runtime_error("Empty batch as input.");
                }
                
                TShapeDickHelper::PickShapeInfo(m_inputShapeStack, p_in);
                
                auto firstInputCont = NSBatchIterLayer::Split<typename TOriInputCont::Keys>(p_in, TOriInputCont::Keys::Create(), 0);
                auto firstOutput = m_kernel.FeedForward(std::move(firstInputCont));
                using OutputKeys = typename decltype(firstOutput)::Keys;
                auto outputCont = NSBatchIterLayer::InitOutputCont<OutputKeys>(std::move(firstOutput), OutputKeys::Create());

                for (size_t i = 1; i < batchNum; ++i)
                {
                    auto curInputCont = NSBatchIterLayer::Split<typename TOriInputCont::Keys>(p_in, TOriInputCont::Keys::Create(), i);
                    auto curOutput = m_kernel.FeedForward(std::move(curInputCont));
                    NSBatchIterLayer::FillOutputCont<OutputKeys>(std::move(curOutput), outputCont);
                }
                return std::move(outputCont);
            }
        }

        template <typename TIn>
        auto FeedBackward(TIn&& p_in)
        {
            using TOriInputCont = RemConstRef<TIn>;
            if constexpr (IsTrivalLayer)
            {
                static_assert(NSBatchIterLayer::IsFBContNonBatch_<typename TOriInputCont::Keys, TOriInputCont>::value);
                return m_kernel.FeedBackward(p_in);
            }
            else if constexpr ((!IsFeedbackOutput) && (!IsUpdate))
            {
                return LayerInputCont<BatchIterLayer>();
            }
            else if constexpr (!IsFeedbackOutput)
            {// Note: update internal parameter
                static_assert(NSBatchIterLayer::IsFBContAllBatch_<typename TOriInputCont::Keys, TOriInputCont>::value, "All grads should be batch.");

                const size_t batchNum = NSBatchIterLayer::GetBatchNum<typename TOriInputCont::Keys>(p_in);
                if (batchNum == 0)
                {
                    throw std::runtime_error("Empty batch as grad input.");
                }
                
                for (size_t i = 1; i <= batchNum; ++i)
                {
                    auto curInputCont = NSBatchIterLayer::Split<typename TOriInputCont::Keys>(p_in, TOriInputCont::Keys::Create(), batchNum - i);
                    m_kernel.FeedBackward(std::move(curInputCont));
                }
            }
            else
            {
                static_assert(NSBatchIterLayer::IsFBContAllBatch_<typename TOriInputCont::Keys, TOriInputCont>::value, "All grads should be batch.");

                const size_t batchNum = NSBatchIterLayer::GetBatchNum<typename TOriInputCont::Keys>(p_in);
                if (batchNum == 0)
                {
                    throw std::runtime_error("Empty batch as grad input.");
                }
                
                auto firstInputGrad = NSBatchIterLayer::Split<typename TOriInputCont::Keys>(p_in, TOriInputCont::Keys::Create(), batchNum - 1);
                auto firstOutputGrad = m_kernel.FeedBackward(std::move(firstInputGrad));
                using OutputGradKeys = typename decltype(firstOutputGrad)::Keys;
                auto outputGrad = NSBatchIterLayer::InitOutputCont<OutputGradKeys>(std::move(firstOutputGrad), OutputGradKeys::Create());
                for (size_t i = 2; i <= batchNum; ++i)
                {
                    auto curInputCont = NSBatchIterLayer::Split<typename TOriInputCont::Keys>(p_in, TOriInputCont::Keys::Create(), batchNum - i);
                    auto curOutput = m_kernel.FeedBackward(std::move(curInputCont));
                    NSBatchIterLayer::FillOutputCont<OutputGradKeys>(std::move(curOutput), outputGrad);
                }

                NSBatchIterLayer::ReverseOutputCont<OutputGradKeys>(outputGrad);                
                return TShapeDickHelper::Collapse(m_inputShapeStack, std::move(outputGrad));
            }
        }
    private:
        std::string m_name;
        KernelType m_kernel;

        typename TShapeDickHelper::type m_inputShapeStack;
    };
}