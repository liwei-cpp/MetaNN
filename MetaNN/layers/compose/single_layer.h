#pragma once

#include <MetaNN/layers/compose/compose_kernel.h>
#include <MetaNN/layers/elementary/weight_layer.h>
#include <MetaNN/layers/elementary/bias_layer.h>
#include <MetaNN/layers/elementary/tanh_layer.h>
#include <MetaNN/layers/elementary/sigmoid_layer.h>

namespace MetaNN
{
template <typename TPolicies> class SingleLayer;

template <>
struct Sublayerof<SingleLayer>
{
    struct Weight;
    struct Bias;
    struct Action;
};

namespace NSSingleLayer
{
using WeightSublayer = Sublayerof<SingleLayer>::Weight;
using BiasSublayer   = Sublayerof<SingleLayer>::Bias;
using ActionSublayer = Sublayerof<SingleLayer>::Action;

template <typename TPlainPolicies>
constexpr static bool HasBias = PolicySelect<SingleLayerPolicy, TPlainPolicies>::HasBias;

template <typename TPlainPolicies> struct ActPick_;

template <>
struct ActPick_<SingleLayerPolicy::ActionTypeCate::Sigmoid>
{
    template <typename T>
    using type = SigmoidLayer<T>;
};

template <>
struct ActPick_<SingleLayerPolicy::ActionTypeCate::Tanh>
{
    template <typename T>
    using type = TanhLayer<T>;
};

template <bool hasBias, template <typename> class TActLayer>
struct TopologyHelper
{
    using type = ComposeTopology<SubLayer<WeightSublayer, WeightLayer>,
                                 SubLayer<BiasSublayer, BiasLayer>,
                                 SubLayer<ActionSublayer, TActLayer>,
                                 InConnect<LayerIO, WeightSublayer, LayerIO>,
                                 InternalConnect<WeightSublayer, LayerIO, BiasSublayer, LayerIO>,
                                 InternalConnect<BiasSublayer, LayerIO, ActionSublayer, LayerIO>,
                                 OutConnect<ActionSublayer, LayerIO, LayerIO>>;
};

template <template <typename> class TActLayer>
struct TopologyHelper<false, TActLayer>
{
    using type = ComposeTopology<SubLayer<WeightSublayer, WeightLayer>,
                                 SubLayer<ActionSublayer, TActLayer>,
                                 InConnect<LayerIO, WeightSublayer, LayerIO>,
                                 InternalConnect<WeightSublayer, LayerIO, ActionSublayer, LayerIO>,
                                 OutConnect<ActionSublayer, LayerIO, LayerIO>>;
};

template <typename TPolicies>
struct KernelHelper
{
    using PlainPolicies = PlainPolicy<TPolicies>;
    constexpr static bool hasBias = HasBias<PlainPolicies>;

    template <typename T>
    using ActType = typename ActPick_<typename PolicySelect<SingleLayerPolicy,
                                                            PlainPolicies>::Action>::template type<T>;

    using type = typename TopologyHelper<hasBias, ActType>::type;
};

template <typename TPolicies>
using Kernel = typename KernelHelper<TPolicies>::type;

template <typename TPolicies>
using Base = ComposeKernel<LayerIO, LayerIO, TPolicies, Kernel<TPolicies>>;

template <bool HasBias, typename TBase>
auto TupleCreator(const std::string& p_name, size_t p_inputLen, size_t p_outputLen)
{
    if constexpr (HasBias)
    {
        return TBase::CreateSubLayers()
                    .template Set<NSSingleLayer::WeightSublayer>(p_name + "-weight", p_inputLen, p_outputLen)
                    .template Set<NSSingleLayer::BiasSublayer>(p_name + "-bias", p_outputLen)
                    .template Set<NSSingleLayer::ActionSublayer>();
    }
    else
    {
        return TBase::CreateSubLayers()
                    .template Set<NSSingleLayer::WeightSublayer>(p_name + "-weight", p_inputLen, p_outputLen)
                    .template Set<NSSingleLayer::ActionSublayer>();
    }
}
}

template <typename TPolicies>
class SingleLayer : public NSSingleLayer::Base<TPolicies>
{
    using TBase = NSSingleLayer::Base<TPolicies>;
public:
    SingleLayer(const std::string& p_name, size_t p_inputLen, size_t p_outputLen)
        : TBase(NSSingleLayer::TupleCreator<NSSingleLayer::KernelHelper<TPolicies>::hasBias, TBase>(p_name, p_inputLen, p_outputLen))
    { }
};
}
