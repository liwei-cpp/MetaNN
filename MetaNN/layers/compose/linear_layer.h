#pragma once

#include <MetaNN/layers/compose/compose_kernel.h>
#include <MetaNN/layers/elementary/bias_layer.h>
#include <MetaNN/layers/elementary/weight_layer.h>

namespace MetaNN
{
template <typename TPolicies> class LinearLayer;

template <>
struct Sublayerof<LinearLayer>
{
    struct Weight;
    struct Bias;
};

namespace NSLinearLayer
{
    using WeightSublayer = Sublayerof<LinearLayer>::Weight;
    using BiasSublayer = Sublayerof<LinearLayer>::Bias;

    using Topology = ComposeTopology<SubLayer<WeightSublayer, WeightLayer>,
                                     SubLayer<BiasSublayer, BiasLayer>,
                                     InConnect<LayerIO, WeightSublayer, LayerIO>,
                                     InternalConnect<WeightSublayer, LayerIO, BiasSublayer, LayerIO>,
                                     OutConnect<BiasSublayer, LayerIO, LayerIO>>;

    template <typename TPolicies>
    using Base = ComposeKernel<LayerIO, LayerIO, TPolicies, Topology>;
}


template <typename TPolicies>
class LinearLayer : public NSLinearLayer::Base<TPolicies>
{
    using TBase = NSLinearLayer::Base<TPolicies>;

public:
    LinearLayer(const std::string& p_name, size_t p_inputLen, size_t p_outputLen)
        : TBase(TBase::CreateSubLayers()
                        .template Set<NSLinearLayer::WeightSublayer>(p_name + "-weight", p_inputLen, p_outputLen)
                        .template Set<NSLinearLayer::BiasSublayer>(p_name + "-bias", p_outputLen))
    { }
};
}
