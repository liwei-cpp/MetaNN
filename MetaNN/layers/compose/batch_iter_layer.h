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

        static constexpr bool IsTrivalLayer = std::is_same_v<TInputs, NullParameter> ? false : NSBatchIterLayer::IsIOMapNonBatch_<TInputs>::value;

    public:
        template <typename... TParams>
        BatchIterLayer(std::string p_name, TParams&&... kernelParams)
            : m_name(std::move(p_name))
            , m_kernel(std::forward<TParams>(kernelParams)...)
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
        }
    
        template <typename TIn>
        auto FeedForward(TIn&& p_in)
        {
            using TOriInputCont = RemConstRef<TIn>;
            if constexpr (IsTrivalLayer)
            {
                static_assert(NSBatchIterLayer::IsFBContNonBatch_<typename TOriInputCont::Keys, TOriInputCont>::value);
                return m_kernel.FeedForward<std::forward<TIn>>(p_in);
            }
            else
            {
                if constexpr (IsEmptyLayerIOMap<InputPortSet> && (IsFeedbackOutput || IsUpdate))
                {
                    // if true, we have no input type info, therefore the input should required be batch for BP.
                    static_assert(NSBatchIterLayer::IsFBContAllBatch_<typename TOriInputCont::Keys, TOriInputCont>::value, "All inputs should be batch.");
                }
                using KernelOutputCont = decltype(std::declval<KernelType>().FeedForward(std::declval<TIn>()));

                const size_t batchNum = NSBatchIterLayer::GetBatchNum<typename TOriInputCont::Keys>(p_in);
                if (batchNum == 0)
                {
                    throw std::runtime_error("Empty batch as input.");
                }
                static_assert(DependencyFalse<TIn>, "Not implemented");
            }
        }

        template <typename TIn>
        auto FeedBackward(TIn&& p_in)
        {
            using TOriInputCont = RemConstRef<TIn>;
            if constexpr (IsTrivalLayer)
            {
                static_assert(NSBatchIterLayer::IsFBContNonBatch_<typename TOriInputCont::Keys, TOriInputCont>::value);
                return m_kernel.FeedBackward<std::forward<TIn>>(p_in);
            }
            else
            {
                static_assert(NSBatchIterLayer::IsFBContAllBatch_<typename TOriInputCont::Keys, TOriInputCont>::value, "All grads should be batch.");
                using KernelOutputCont = decltype(std::declval<KernelType>().FeedBackward(std::declval<TIn>()));

                const size_t batchNum = NSBatchIterLayer::GetBatchNum<typename TOriInputCont::Keys>(p_in);
                if (batchNum == 0)
                {
                    throw std::runtime_error("Empty batch as grad input.");
                }
                static_assert(DependencyFalse<TIn>, "Not implemented");
            }
        }
    private:
        std::string m_name;
        KernelType m_kernel;
    };
}