#pragma once

#include <MetaNN/layers/composite/compose_kernel.h>
#include <MetaNN/layers/principal/_.h>

namespace MetaNN
{
    struct WeightParamSublayer;
    struct BiasParamSublayer;
    struct AddSublayer;
    struct DotSublayer;
    
    struct LayerInput; struct LayerOutput;
    namespace NSLinearLayer
    {
        using Topology = ComposeTopology<Sublayer<WeightParamSublayer, ParamSourceLayer>,
                                         Sublayer<BiasParamSublayer, ParamSourceLayer>,
                                         Sublayer<AddSublayer, AddLayer>,
                                         Sublayer<DotSublayer, DotLayer>,
                                         InConnect<LayerInput, DotSublayer, LeftOperand>,
                                         InternalConnect<WeightParamSublayer, LayerOutput, DotSublayer, RightOperand>,
                                         InternalConnect<DotSublayer, LayerOutput, AddSublayer, LeftOperand>,
                                         InternalConnect<BiasParamSublayer, LayerOutput, AddSublayer, RightOperand>,
                                         OutConnect<AddSublayer, LayerOutput, LayerOutput>>;

        template <typename TInputMap, typename TPolicies>
        using Base = ComposeKernel<LayerPortSet<LayerInput>, LayerPortSet<LayerOutput>, TInputMap, TPolicies, Topology>;
    }
    
    template <typename TInputs, typename TPolicies>
    class LinearLayer : public NSLinearLayer::Base<TInputs, TPolicies>
    {
        using TBase = NSLinearLayer::Base<TInputs, TPolicies>;
        using WeightParamType = typename TBase::template SublayerType<WeightParamSublayer>::ParamType;
        using BiasParamType = typename TBase::template SublayerType<BiasParamSublayer>::ParamType;
        using WeightParamShapeType = RemConstRef<decltype(std::declval<WeightParamType>().Shape())>;
        using BiasParamShapeType = RemConstRef<decltype(std::declval<BiasParamType>().Shape())>;

    public:
        LinearLayer(const std::string& p_name, WeightParamShapeType weightShape, BiasParamShapeType biasShape)
            : TBase(TBase::CreateSublayers()
                        .template Set<WeightParamSublayer>(p_name + "/weight", std::move(weightShape))
                        .template Set<BiasParamSublayer>(p_name + "/bias", std::move(biasShape))
                        .template Set<AddSublayer>(p_name + "/add")
                        .template Set<DotSublayer>(p_name + "/dot"))
        { }
    };
}
