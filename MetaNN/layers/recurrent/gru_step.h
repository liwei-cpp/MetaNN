#pragma once
#include <MetaNN/layers/composite/compose_kernel.h>
#include <MetaNN/layers/composite/weight_layer.h>
#include <MetaNN/layers/principal/_.h>

namespace MetaNN
{
    struct LayerInput; struct LayerOutput;
    namespace NSGruStep
    {
        struct Wz; struct Uz; struct Add_z; struct Act_z; 
        struct Wr; struct Ur; struct Add_r; struct Act_r;
        struct W; struct U; struct Mult; struct Add; struct Act_Hat;
        struct Interpolate;
        
        using Topology = ComposeTopology<
            Sublayer<Wz, WeightLayer>,
            Sublayer<Uz, WeightLayer>,
            Sublayer<Add_z, AddLayer>,
            Sublayer<Act_z, SigmoidLayer>,
            InConnect<LayerInput, Wz, LayerInput>,
            InConnect<Previous<LayerOutput>, Uz, LayerInput>,
            InternalConnect<Wz, LayerOutput, Add_z, LeftOperand>,
            InternalConnect<Uz, LayerOutput, Add_z, RightOperand>,
            InternalConnect<Add_z, LayerOutput, Act_z, LayerInput>,

            Sublayer<Wr, WeightLayer>,
            Sublayer<Ur, WeightLayer>,
            Sublayer<Add_r, AddLayer>,
            Sublayer<Act_r, SigmoidLayer>,
            InConnect<LayerInput, Wr, LayerInput>,
            InConnect<Previous<LayerOutput>, Ur, LayerInput>,
            InternalConnect<Wr, LayerOutput, Add_r, LeftOperand>,
            InternalConnect<Ur, LayerOutput, Add_r, RightOperand>,
            InternalConnect<Add_r, LayerOutput, Act_r, LayerInput>,

            Sublayer<Mult, MultiplyLayer>,
            InConnect<Previous<LayerOutput>, Mult, LeftOperand>,
            InternalConnect<Act_r, LayerOutput, Mult, RightOperand>,
            Sublayer<U, WeightLayer>,
            InternalConnect<Mult, LayerOutput, U, LayerInput>,
            Sublayer<W, WeightLayer>,
            InConnect<LayerInput, W, LayerInput>,
            Sublayer<Add, AddLayer>,
            InternalConnect<U, LayerOutput, Add, LeftOperand>,
            InternalConnect<W, LayerOutput, Add, RightOperand>,
            Sublayer<Act_Hat, TanhLayer>,
            InternalConnect<Add, LayerOutput, Act_Hat, LayerInput>,

            Sublayer<Interpolate, InterpolateLayer>,
            InternalConnect<Act_Hat, LayerOutput, Interpolate, InterpolateLayerWeight1>,
            InConnect<Previous<LayerOutput>, Interpolate, InterpolateLayerWeight2>,
            InternalConnect<Act_z, LayerOutput, Interpolate, InterpolateLayerLambda>,

            OutConnect<Interpolate, LayerOutput, LayerOutput>>;

        template <typename TInputMap, typename TPolicies>
        using Base = ComposeKernel<LayerPortSet<LayerInput, Previous<LayerOutput>>,
                                   LayerPortSet<LayerOutput>, TInputMap, TPolicies, Topology>;
        
        template <typename TPolicies>
        struct CalParameterPolicy_
        {
            static_assert(IsPolicyContainer<TPolicies>, "TPolicies is not a policy container.");
            using CurLayerPolicy = PlainPolicy<TPolicies>;
            using ElementType = typename PolicySelect<ParamPolicy, CurLayerPolicy>::ElementType;
            using DeviceType = typename PolicySelect<ParamPolicy, CurLayerPolicy>::DeviceType;

            using type = ChangePolicy<PParamTypeIs<Matrix<ElementType, DeviceType>>, TPolicies>;
        };
        
        template <typename TPolicies>
        using CalParameterPolicy = typename CalParameterPolicy_<TPolicies>::type;
    }
    
    template <typename TInputMap, typename TPolicies>
    class GruStep : public NSGruStep::Base<TInputMap, NSGruStep::CalParameterPolicy<TPolicies>>
    {
        using ModifiedPolicy = NSGruStep::CalParameterPolicy<TPolicies>;
        using TBase = NSGruStep::Base<TInputMap, ModifiedPolicy>;

    public:
        GruStep(const std::string& p_name, size_t p_fanIn, size_t p_fanOut)
            : TBase(TBase::CreateSublayers()
                        .template Set<NSGruStep::Wz>(p_name + "/Wz", p_fanIn, p_fanOut)
                        .template Set<NSGruStep::Uz>(p_name + "/Uz", p_fanOut, p_fanOut)
                        .template Set<NSGruStep::Add_z>(p_name + "/Add_z")
                        .template Set<NSGruStep::Act_z>(p_name + "/Act_z")
                        .template Set<NSGruStep::Wr>(p_name + "/Wr", p_fanIn, p_fanOut)
                        .template Set<NSGruStep::Ur>(p_name + "/Ur", p_fanOut, p_fanOut)
                        .template Set<NSGruStep::Add_r>(p_name + "/Add_r")
                        .template Set<NSGruStep::Act_r>(p_name + "/Act_r")
                        .template Set<NSGruStep::W>(p_name + "/W", p_fanIn, p_fanOut)
                        .template Set<NSGruStep::U>(p_name + "/U", p_fanOut, p_fanOut)
                        .template Set<NSGruStep::Mult>(p_name + "/Mult")
                        .template Set<NSGruStep::Add>(p_name + "/Add")
                        .template Set<NSGruStep::Act_Hat>(p_name + "/Act_Hat")
                        .template Set<NSGruStep::Interpolate>(p_name + "/Interpolate"))
        {}
    };
}