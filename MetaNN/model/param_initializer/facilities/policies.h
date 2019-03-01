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
}

#include <MetaNN/policies/policy_macro_end.h>