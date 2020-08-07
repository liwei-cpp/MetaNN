#pragma once

#include <MetaNN/layers/composite/compose_kernel.h>
#include <MetaNN/layers/principal/_.h>

namespace MetaNN
{
    struct ParamSublayer;
    struct DotSublayer;
    
    struct LayerInput; struct LayerOutput;
    namespace NSWeightLayer
    {
        using Topology = ComposeTopology<Sublayer<ParamSublayer, ParamSourceLayer>,
                                         Sublayer<DotSublayer, DotLayer>,
                                         InConnect<LayerInput, DotSublayer, LeftOperand>,
                                         InternalConnect<ParamSublayer, LayerOutput, DotSublayer, RightOperand>,
                                         OutConnect<DotSublayer, LayerOutput, LayerOutput>>;

        template <typename TInputs, typename TPolicies>
        using Base = ComposeKernel<LayerPortSet<LayerInput>, LayerPortSet<LayerOutput>, TInputs, TPolicies, Topology>;
    }
    
    template <typename TInputs, typename TPolicies>
    class WeightLayer : public NSWeightLayer::Base<TInputs, TPolicies>
    {
        using TBase = NSWeightLayer::Base<TInputs, TPolicies>;

    public:
        template <typename... TShapeParams>
        WeightLayer(const std::string& p_name, TShapeParams&&... shapeParams)
            : TBase(TBase::CreateSublayers()
                        .template Set<ParamSublayer>(p_name + "/param", std::forward<TShapeParams>(shapeParams)...)
                        .template Set<DotSublayer>(p_name + "/add"))
        { }
    };
}