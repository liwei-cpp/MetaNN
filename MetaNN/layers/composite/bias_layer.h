#pragma once

#include <MetaNN/layers/composite/compose_kernel.h>
#include <MetaNN/layers/principal/_.h>

namespace MetaNN
{
    struct ParamSublayer;
    struct AddSublayer;
    
    struct LayerInput; struct LayerOutput;
    namespace NSBiasLayer
    {
        using Topology = ComposeTopology<Sublayer<ParamSublayer, ParamSourceLayer>,
                                         Sublayer<AddSublayer, AddLayer>,
                                         InConnect<LayerInput, AddSublayer, LeftOperand>,
                                         InternalConnect<ParamSublayer, LayerOutput, AddSublayer, RightOperand>,
                                         OutConnect<AddSublayer, LayerOutput, LayerOutput>>;

        template <typename TInputMap, typename TPolicies>
        using Base = ComposeKernel<LayerPortSet<LayerInput>, LayerPortSet<LayerOutput>, TInputMap, TPolicies, Topology>;
    }
    
    template <typename TInputs, typename TPolicies>
    class BiasLayer : public NSBiasLayer::Base<TInputs, TPolicies>
    {
        using TBase = NSBiasLayer::Base<TInputs, TPolicies>;

    public:
        template <typename... TShapeParams>
        BiasLayer(const std::string& p_name, TShapeParams&&... shapeParams)
            : TBase(TBase::CreateSublayers()
                        .template Set<ParamSublayer>(p_name + "/param", std::forward<TShapeParams>(shapeParams)...)
                        .template Set<AddSublayer>(p_name + "/add"))
        { }
    };
}