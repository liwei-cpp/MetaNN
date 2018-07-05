#pragma once

#include <MetaNN/policies/policy_macro_begin.h>
#include <MetaNN/data/facilities/tags.h>
namespace MetaNN
{
struct FeedbackPolicy
{
    using MajorClass = FeedbackPolicy;
    
    struct IsUpdateValueCate;
    struct IsFeedbackOutputValueCate;

    static constexpr bool IsUpdate = false;
    static constexpr bool IsFeedbackOutput = false;
};
ValuePolicyObj(PUpdate,   FeedbackPolicy, IsUpdate, true);
ValuePolicyObj(PNoUpdate, FeedbackPolicy, IsUpdate, false);
ValuePolicyObj(PFeedbackOutput,   FeedbackPolicy, IsFeedbackOutput, true);
ValuePolicyObj(PFeedbackNoOutput, FeedbackPolicy, IsFeedbackOutput, false);

struct InputPolicy
{
    using MajorClass = InputPolicy;
    
    struct BatchModeValueCate;
    static constexpr bool BatchMode = false;
};
ValuePolicyObj(PBatchMode,  InputPolicy, BatchMode, true);
ValuePolicyObj(PNoBatchMode,InputPolicy, BatchMode, false);

struct OperandPolicy
{
    using MajorClass = OperandPolicy;
    
    struct DeviceTypeCate : public MetaNN::DeviceTags {};
    using Device = DeviceTypeCate::CPU;
    
    struct ElementTypeCate;
    using Element = float;
};
TypePolicyObj(PCPUDevice, OperandPolicy, Device, CPU);
TypePolicyTemplate(PElementTypeIs, OperandPolicy, Element);

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
