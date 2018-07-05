#pragma once

#include <type_traits>
namespace MetaNN
{
template <typename TLayer, typename TIn>
auto LayerFeedForward(TLayer& layer, TIn&& p_in)
{
    return layer.FeedForward(std::forward<TIn>(p_in));
}

template <typename TLayer, typename TGrad>
auto LayerFeedBackward(TLayer& layer, TGrad&& p_grad)
{
    return layer.FeedBackward(std::forward<TGrad>(p_grad));
}

/// init interface ========================================
namespace NSLayerInterface
{
template <typename L, typename TInitializer, typename TBuffer, typename TInitPolicies>
std::true_type InitTest(decltype(&L::template Init<TInitializer, TBuffer, TInitPolicies>));

template <typename L, typename TInitPolicies, typename TInitContainer, typename TLoad>
std::false_type InitTest(...);

template <typename L, typename TGradCollector>
std::true_type GradCollectTest(decltype(&L::template GradCollect<TGradCollector>));

template <typename L, typename TGradCollector>
std::false_type GradCollectTest(...);

template <typename L, typename TSave>
std::true_type SaveWeightsTest(decltype(&L::template SaveWeights<TSave>));

template <typename L, typename TSave>
std::false_type SaveWeightsTest(...);

template <typename L>
std::true_type NeutralInvariantTest(decltype(&L::NeutralInvariant));

template <typename L>
std::false_type NeutralInvariantTest(...);
}

template <typename TLayer, typename TInitializer, typename TBuffer, 
          typename TInitPolicies = typename TInitializer::PolicyCont>
void LayerInit(TLayer& layer, TInitializer& initializer, TBuffer& loadBuffer, std::ostream* log = nullptr)
{
    if constexpr (decltype(NSLayerInterface::InitTest<TLayer, TInitializer, TBuffer, TInitPolicies>(nullptr))::value)
        layer.template Init<TInitializer, TBuffer, TInitPolicies>(initializer, loadBuffer, log);
}
    
template <typename TLayer, typename TGradCollector>
void LayerGradCollect(TLayer& layer, TGradCollector& gc)
{
    if constexpr(decltype(NSLayerInterface::GradCollectTest<TLayer, TGradCollector>(nullptr))::value)
        layer.GradCollect(gc);
}

template <typename TLayer, typename TSave>
void LayerSaveWeights(const TLayer& layer, TSave& saver)
{
    if constexpr (decltype(NSLayerInterface::SaveWeightsTest<TLayer, TSave>(nullptr))::value)
        layer.SaveWeights(saver);
}

template <typename TLayer>
void LayerNeutralInvariant(TLayer& layer)
{
    if constexpr (decltype(NSLayerInterface::NeutralInvariantTest<TLayer>(nullptr))::value)
        layer.NeutralInvariant();
}
}
