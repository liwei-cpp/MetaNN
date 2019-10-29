#pragma once
/*
#include <MetaNN/layers/recurrent/gru_step.h>
#include <cassert>

namespace MetaNN
{
namespace NSRecurrentLayer
{
template <typename TStep, typename TPolicy> struct StepEnum2Type_;

template <typename TPolicy>
struct StepEnum2Type_<RecurrentLayerPolicy::StepTypeCate::GRU, TPolicy>
{
    using type = GruStep<TPolicy>;
};

template <typename TStep, typename TPolicy>
using StepEnum2Type = typename StepEnum2Type_<TStep, TPolicy>::type;
}

template <typename TPolicies>
class RecurrentLayer
{
    static_assert(IsPolicyContainer<TPolicies>, "TPolicies is not policy container.");
    using CurLayerPolicy = PlainPolicy<TPolicies>;

public:
    static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, TPolicies>::IsFeedbackOutput;
    static constexpr bool IsUpdate = PolicySelect<FeedbackPolicy, TPolicies>::IsUpdate;

private:
    static constexpr bool UseBptt = PolicySelect<RecurrentLayerPolicy, CurLayerPolicy>::UseBptt;

    using StepPolicy = typename std::conditional_t<(!IsFeedbackOutput) && IsUpdate && UseBptt,
                                                   ChangePolicy_<PFeedbackOutput, TPolicies>,
                                                   Identity_<TPolicies>>::type;

    using StepEnum = typename PolicySelect<RecurrentLayerPolicy, CurLayerPolicy>::Step;
    using StepType = NSRecurrentLayer::StepEnum2Type<StepEnum, StepPolicy>;

    using ElementType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::Element;
    using DeviceType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::Device;

    constexpr static bool m_BatchMode = PolicySelect<InputPolicy, CurLayerPolicy>::BatchMode;
    using DataType = std::conditional_t<m_BatchMode, 
                                        DynamicData<ElementType, DeviceType, CategoryTags::BatchMatrix>,
                                        DynamicData<ElementType, DeviceType, CategoryTags::Matrix>>;

public:
    using InputType = typename StepType::InputType;
    using OutputType = typename StepType::OutputType;

public:
    template <typename...T>
    RecurrentLayer(T&&... params)
        : m_step(std::forward<T>(params)...)
        , m_inForward(true)
    {}

public:
    template <typename TInitializer, typename TBuffer, 
              typename TInitPolicies = typename TInitializer::PolicyCont>
    void Init(TInitializer& initializer, TBuffer& loadBuffer, std::ostream* log = nullptr)
    {
        m_step.template Init<TInitializer, TBuffer, TInitPolicies>(initializer, loadBuffer, log);
    }

    template <typename TSave>
    void SaveWeights(TSave& saver)
    {
        m_step.SaveWeights(saver);
    }

    template <typename TGradCollector>
    void GradCollect(TGradCollector& col)
    {
        m_step.GradCollect(col);
    }

    template <typename TIn>
    auto FeedForward(TIn&& p_in)
    {
        auto& init = p_in.template Get<RnnLayerHiddenBefore>();
        using rawType = std::decay_t<decltype(init)>;
        m_inForward = true;

        if constexpr(std::is_same<rawType, NullParameter>::value)
        {
            assert(!m_hiddens.IsEmpty());
            auto real_in = std::move(p_in).template Set<RnnLayerHiddenBefore>(m_hiddens);
            auto res = m_step.FeedForward(std::move(real_in));
            m_hiddens = MakeDynamic(res.template Get<LayerIO>());
            return res;
        }
        else
        {
            auto res = m_step.FeedForward(std::forward<TIn>(p_in));
            m_hiddens = MakeDynamic(res.template Get<LayerIO>());
            return res;
        }
    }

    template <typename TGrad>
    auto FeedBackward(const TGrad& p_grad)
    {
        if constexpr(UseBptt)
        {
            auto gradVal = p_grad.template Get<LayerIO>();
            if (!m_inForward)
            {
                auto newGrad = MakeDynamic(gradVal + m_hiddens);
                auto input = LayerIO::Create().template Set<LayerIO>(newGrad);
                auto res = m_step.FeedBackward(std::move(input));
                m_hiddens = MakeDynamic(res.template Get<RnnLayerHiddenBefore>());
                return std::move(res);
            }
            else
            {
                m_inForward = false;
                
                auto newGrad = MakeDynamic(gradVal);
                auto input = LayerIO::Create().template Set<LayerIO>(newGrad);
                auto res = m_step.FeedBackward(std::move(input));
                m_hiddens = MakeDynamic(res.template Get<RnnLayerHiddenBefore>());
                return std::move(res);
            }
        }
        else
        {
            return m_step.FeedBackward(std::forward<TGrad>(p_grad));
        }
    }

    void NeutralInvariant()
    {
        m_step.NeutralInvariant();
    }

private:
    StepType m_step;
    DataType m_hiddens;
    bool     m_inForward;
};
}
*/

namespace MetaNN
{
    template <typename TPort> struct Previous;
    
    namespace NSRecurrentLayer
    {
        template <typename T>
        constexpr bool IsBatchCategory = IsBatchCategoryTag<typename T::CategoryTag> || IsBatchSequenceCategoryTag<typename T::CategoryTag>;

        template <typename TInputSet, typename TOutputSet>
        struct NoIOPortOverLap_;
        
        template <typename TInputSet, typename... TOutputPorts>
        struct NoIOPortOverLap_<TInputSet, LayerPortSet<TOutputPorts...>>
        {
            constexpr static bool value = !(Set::HasKey<TInputSet, Previous<TOutputPorts>> || ...);
        };
        
        template <typename TInputSet, typename TOutputSet>
        struct MergeIOPortSet_;
        
        template <typename... TInputPorts, typename... TOutputPorts>
        struct MergeIOPortSet_<LayerPortSet<TInputPorts...>, LayerPortSet<TOutputPorts...>>
        {
            using type = LayerPortSet<TInputPorts..., Previous<TOutputPorts>...>;
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
            
            using OutputPortSet = typename KernelType::OutputPortSet;
            static_assert(NoIOPortOverLap_<typename KernelType::InputPortSet, OutputPortSet>::value);
            using InputPortSet = typename MergeIOPortSet_<typename KernelType::InputPortSet, OutputPortSet>::type;
            
            using InputMap = EmptyLayerIOMap_<InputPortSet>;
            
            constexpr static bool IsTrival = false;
        };
        
        template <typename TKey>
        constexpr bool IsPreviousPort = false;
        
        template <typename TKey>
        constexpr bool IsPreviousPort<Previous<TKey>> = true;
        
        template <typename... TKeys, typename... TValues, typename TPolicies>
        struct KernelGenerator_<LayerIOMap<LayerKV<TKeys, TValues>...>, TPolicies>
        {
            static_assert((!IsBatchCategory<TValues> && ...), "No batch input is allowed in RNN layer.");
            static_assert((IsPreviousPort<TKeys> || ...), "No Previous port in the input port set.");
            static_assert(!((IsPreviousPort<TKeys> && IsSequenceCategoryTag<typename TValues::CategoryTag>) || ...),
                          "Previous ports should not be sequence.");

            constexpr static bool IsTrival = !(IsSequenceCategoryTag<typename TValues::CategoryTag> || ...);
            
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

            using KernelType = Kernel<NullParameter, AmendKernelPolicy>;

            using OutputPortSet = typename KernelType::OutputPortSet;
            static_assert(NoIOPortOverLap_<typename KernelType::InputPortSet, OutputPortSet>::value);
            using InputPortSet = typename MergeIOPortSet_<typename KernelType::InputPortSet, OutputPortSet>::type;
            static_assert(Set::IsEqual<LayerPortSet<TKeys...>, InputPortSet>, "Invalid input port set.");
            
            using InputMap = LayerIOMap<LayerKV<TKeys, TValues>...>;
        };
    }
    
    template <typename TInputs, typename TPolicies>
    class RecurrentLayer
    {
        static_assert(IsPolicyContainer<TPolicies>);
        using KernelGen = NSRecurrentLayer::KernelGenerator_<TInputs, TPolicies>;

        using KernelType = typename KernelGen::KernelType;
    public:
        static constexpr bool IsFeedbackOutput = KernelType::IsFeedbackOutput;
        static constexpr bool IsUpdate = KernelType::IsUpdate;

        using InputPortSet = typename KernelGen::InputPortSet;
        using OutputPortSet = typename KernelGen::OutputPortSet;
        using InputMap = typename KernelGen::InputMap;
        
        constexpr static bool IsTrivalLayer = KernelGen::IsTrival;

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
            LayerNeutralInvariant(m_kernel);
        }

    private:
        std::string m_name;
        KernelType m_kernel;
    };
}