#pragma once

#include <MetaNN/policies/policy_macro_begin.h>
#include <MetaNN/data/facilities/tags.h>
namespace MetaNN
{
struct GradPolicy
{
    using MajorClass = GradPolicy;
    
    struct IsUpdateValueCate;
    struct IsFeedbackOutputValueCate;

    static constexpr bool IsUpdate = false;
    static constexpr bool IsFeedbackOutput = false;
};
ValuePolicyObj(PUpdate,           GradPolicy, IsUpdate, true);
ValuePolicyObj(PNoUpdate,         GradPolicy, IsUpdate, false);
ValuePolicyObj(PFeedbackOutput,   GradPolicy, IsFeedbackOutput, true);
ValuePolicyObj(PFeedbackNoOutput, GradPolicy, IsFeedbackOutput, false);

struct ParamPolicy
{
    using MajorClass = ParamPolicy;
    
    struct ParamTypeCate
    {
        using Scalar = CategoryTags::Scalar;
        using Matrix = CategoryTags::Matrix;
        using ThreeDArray = CategoryTags::ThreeDArray;
    };
    using ParamType = ParamTypeCate::Matrix;
};
TypePolicyObj(PScalarParam,      ParamPolicy, Param, Scalar);
TypePolicyObj(PMatrixParam,      ParamPolicy, Param, Matrix);
TypePolicyObj(PThreeDArrayParam, ParamPolicy, Param, ThreeDArray);

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
