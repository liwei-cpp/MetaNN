#pragma once

#include <random>
#include <MetaNN/policies/policy_macro_begin.h>
namespace MetaNN
{
struct InitPolicy
{
    using MajorClass = InitPolicy;
    
    struct RandEngineTypeCate;
    using RandEngine = std::mt19937;
};

TypePolicyTemplate(PRandomGeneratorIs,   InitPolicy, RandEngine);

struct VarScaleFillerPolicy
{
    using MajorClass = VarScaleFillerPolicy;
    
    struct DistributeTypeCate
    {
        struct Uniform;
        struct Norm;
    };
    using Distribute = DistributeTypeCate::Uniform;
    
    struct ScaleModeTypeCate
    {
        struct FanIn;
        struct FanOut;
        struct FanAvg;
    };
    using ScaleMode = ScaleModeTypeCate::FanAvg;
};
TypePolicyObj(PNormVarScale,    VarScaleFillerPolicy, Distribute, Norm);
TypePolicyObj(PUniformVarScale, VarScaleFillerPolicy, Distribute, Uniform);
TypePolicyObj(PVarScaleFanIn,   VarScaleFillerPolicy, ScaleMode,  FanIn);
TypePolicyObj(PVarScaleFanOut,  VarScaleFillerPolicy, ScaleMode,  FanOut);
TypePolicyObj(PVarScaleFanAvg,  VarScaleFillerPolicy, ScaleMode,  FanAvg);
}

#include <MetaNN/policies/policy_macro_end.h>