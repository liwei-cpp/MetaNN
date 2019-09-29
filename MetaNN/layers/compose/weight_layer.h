#pragma once

#include <MetaNN/layers/compose/compose_kernel.h>
#include <MetaNN/layers/elementary/dot_layer.h>
#include <MetaNN/layers/source/param_source_layer.h>

namespace MetaNN
{
    struct ParamSublayer;
    struct DotSublayer;
    
    namespace NSWeightLayer
    {
        using Topology = ComposeTopology<Sublayer<ParamSublayer, ParamSourceLayer>,
                                         Sublayer<DotSublayer, DotLayer>,
                                         InConnect<LayerInput, DotSublayer, LeftOperand>,
                                         InternalConnect<ParamSublayer, LayerOutput, DotSublayer, RightOperand>,
                                         OutConnect<DotSublayer, LayerOutput, LayerOutput>>;

        template <typename TInSet, typename TOutSet, typename TInputs, typename TPolicies>
        using Base = ComposeKernel<TInSet, TOutSet, TInputs, TPolicies, Topology>;
    }
    
    template <typename TInputs, typename TPolicies>
    class WeightLayer : public NSWeightLayer::Base<LayerInputPortSet<WeightLayer>, LayerOutputPortSet<WeightLayer>, TInputs, TPolicies>
    {
        using TBase = NSWeightLayer::Base<LayerInputPortSet<WeightLayer>, LayerOutputPortSet<WeightLayer>, TInputs, TPolicies>;

    public:
        template <typename... TShapeParams>
        WeightLayer(const std::string& p_name, TShapeParams&&... shapeParams)
            : TBase(TBase::CreateSublayers()
                        .template Set<ParamSublayer>(p_name + "-param", p_name, std::forward<TShapeParams>(shapeParams)...)
                        .template Set<DotSublayer>(p_name + "-add"))
        { }
    };
}