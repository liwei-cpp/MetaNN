#pragma once

#include <MetaNN/layers/compose/compose_kernel.h>
#include <MetaNN/layers/elementary/add_layer.h>
#include <MetaNN/layers/source/param_source_layer.h>

namespace MetaNN
{
/*    struct LayerInput;
    struct LayerOutput;
    
    struct ParameterSublayer;
    struct AddSublayer;
    
    namespace NSBiasLayer
    {
        using Topology = ComposeTopology<Sublayer<ParameterSublayer, ParamSourceLayer>,
                                         Sublayer<AddSublayer, AddLayer>,
                                         InConnect<LayerInput, AddSublayer, LeftOperand>,
                                         InternalConnect<ParameterSublayer, LayerOutput, AddSublayer, RightOperand>,
                                         OutConnect<AddSublayer, LayerOutput, LayerOutput>>;

        template <typename TInputs, typename TGrads, typename TPolicies>
        using Base = ComposeKernel<TInputs, TGrads, TPolicies, Topology>;
    }
    
    template <typename TInputs, typename TGrads, typename TPolicies>
    class BiasLayer : public NSBiasLayer::Base<TInputs, TGrads, TPolicies>
    {
        using TBase = NSBiasLayer::Base<TInputs, TGrads, TPolicies>;

    public:
        template <typename... TShapeParams>
        BiasLayer(const std::string& p_name, TShapeParams&&... shapeParams)
            : TBase(TBase::CreateSublayers()
                        .template Set<ParameterSublayer>(p_name + "-param", std::forward<TShapeParams>(shapeParams)...)
                        .template Set<AddSublayer>(p_name + "-add"))
        { }
    };*/
}