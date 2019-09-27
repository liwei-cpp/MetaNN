#pragma once

#include <MetaNN/layers/compose/compose_kernel.h>
#include <MetaNN/layers/elementary/add_layer.h>
#include <MetaNN/layers/source/param_source_layer.h>

namespace MetaNN
{
    struct ParameterSublayer;
    struct AddSublayer;
    
    namespace NSBiasLayer
    {
        using Topology = ComposeTopology<Sublayer<ParameterSublayer, ParamSourceLayer>,
                                         Sublayer<AddSublayer, AddLayer>,
                                         InConnect<LayerInput, AddSublayer, LeftOperand>,
                                         InternalConnect<ParameterSublayer, LayerOutput, AddSublayer, RightOperand>,
                                         OutConnect<AddSublayer, LayerOutput, LayerOutput>>;

        template <typename TInSet, typename TOutSet, typename TInputs, typename TPolicies>
        using Base = ComposeKernel<TInSet, TOutSet, TInputs, TPolicies, Topology>;
    }
    
    template <typename TInputs, typename TPolicies>
    class BiasLayer : public NSBiasLayer::Base<LayerInputPortSet<BiasLayer>, LayerOutputPortSet<BiasLayer>, TInputs, TPolicies>
    {
        using TBase = NSBiasLayer::Base<LayerInputPortSet<BiasLayer>, LayerOutputPortSet<BiasLayer>, TInputs, TPolicies>;

    public:
        template <typename... TShapeParams>
        BiasLayer(const std::string& p_name, TShapeParams&&... shapeParams)
            : TBase(TBase::CreateSublayers()
                        .template Set<ParameterSublayer>(p_name + "-param", p_name, std::forward<TShapeParams>(shapeParams)...)
                        .template Set<AddSublayer>(p_name + "-add"))
        { }
    };
}