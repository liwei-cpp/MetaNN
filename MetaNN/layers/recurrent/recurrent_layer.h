#pragma once

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
            if (!m_inForward)
            {
                auto gradVal = p_grad.template Get<LayerIO>();
                auto newGrad = LayerIO::Create().template Set<LayerIO>(gradVal + m_hiddens);
                return m_step.FeedStepBackward(newGrad, m_hiddens);
            }
            else
            {
                m_inForward = false;
                return m_step.FeedStepBackward(p_grad, m_hiddens);
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
