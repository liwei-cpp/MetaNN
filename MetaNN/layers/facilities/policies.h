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

    struct ParamPolicy
    {
        using MajorClass = ParamPolicy;
        
        struct ParamTypeTypeCate;
        using ParamType = NullParameter;

        struct InitializerTypeCate;
        using Initializer = NullParameter;
    };
    TypePolicyTemplate(PParamTypeIs,   ParamPolicy, ParamType);
    TypePolicyTemplate(PInitializerIs, ParamPolicy, Initializer);
    
    struct LayerStructurePolicy
    {
        using MajorClass = LayerStructurePolicy;
        // ActFunc
        struct ActFuncTemplateCate;
        
        template <typename TInputMap, typename TPolicies>
        struct DummyActFun;

        template <typename TInputMap, typename TPolicies>
        using ActFunc = DummyActFun<TInputMap, TPolicies>;
        
        template <template<typename, typename> class T>
        static constexpr bool IsDummyActFun = std::is_same_v<T<void, void>, DummyActFun<void, void>>;
        
        // Bias Involved
        struct BiasInvolvedValueCate;
        static constexpr bool BiasInvolved = true;
    };

    template <template <typename, typename> class T>
    struct PActFuncIs : virtual public LayerStructurePolicy
    {
        using MinorClass = LayerStructurePolicy::ActFuncTemplateCate;
        
        template <typename TInputMap, typename TPolicies>
        using ActFunc = T<TInputMap, TPolicies>;
    };
    ValuePolicyObj(PBiasInvolved,    LayerStructurePolicy, BiasInvolved, true);
    ValuePolicyObj(PBiasNotInvolved, LayerStructurePolicy, BiasInvolved, false);
}
#include <MetaNN/policies/policy_macro_end.h>
