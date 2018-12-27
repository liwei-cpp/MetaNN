#pragma once

#include <MetaNN/policies/policy_macro_begin.h>
#include <MetaNN/data/facilities/tags.h>
namespace MetaNN
{
struct GeneralPolicy
{
    using MajorClass = GeneralPolicy;
    
    struct IsUpdateValueCate;
    struct IsFeedbackOutputValueCate;

    static constexpr bool IsUpdate = false;
    static constexpr bool IsFeedbackOutput = false;
};
ValuePolicyObj(PUpdate,           GeneralPolicy, IsUpdate, true);
ValuePolicyObj(PNoUpdate,         GeneralPolicy, IsUpdate, false);
ValuePolicyObj(PFeedbackOutput,   GeneralPolicy, IsFeedbackOutput, true);
ValuePolicyObj(PFeedbackNoOutput, GeneralPolicy, IsFeedbackOutput, false);

struct SingleLayerPolicy
{
    using MajorClass = SingleLayerPolicy;
    
    struct ActionTypeCate
    {
        struct Sigmoid;
        struct Tanh;
    };
    struct HasBiasValueCate;

    using Action = ActionTypeCate::Sigmoid;
    static constexpr bool HasBias = true;
};
TypePolicyObj(PSigmoidAction, SingleLayerPolicy, Action, Sigmoid);
TypePolicyObj(PTanhAction, SingleLayerPolicy, Action, Tanh);
ValuePolicyObj(PBiasSingleLayer,  SingleLayerPolicy, HasBias, true);
ValuePolicyObj(PNoBiasSingleLayer, SingleLayerPolicy, HasBias, false);

struct RecurrentLayerPolicy
{
    using MajorClass = RecurrentLayerPolicy;
    
    struct StepTypeCate
    {
        struct GRU;
    };
    struct UseBpttValueCate;

    using Step = StepTypeCate::GRU;
    constexpr static bool UseBptt = true;
};
TypePolicyObj(PRecGRUStep, RecurrentLayerPolicy, Step, GRU);
ValuePolicyObj(PEnableBptt,  RecurrentLayerPolicy, UseBptt, true);
ValuePolicyObj(PDisableBptt,  RecurrentLayerPolicy, UseBptt, false);
}
#include <MetaNN/policies/policy_macro_end.h>
