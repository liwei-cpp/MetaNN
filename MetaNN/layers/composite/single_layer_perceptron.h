#pragma once

#include <MetaNN/layers/composite/compose_kernel.h>
#include <MetaNN/layers/composite/linear_layer.h>
#include <MetaNN/layers/principal/_.h>

namespace MetaNN
{
    struct WeightParamSublayer;
    struct DotSublayer;
    struct LinearSublayer;
    struct ActivateSublayer;
    
    template <typename TInputs, typename TPolicies>
    class SingleLayerPerceptron;
    
    namespace NSSingleLayerPerceptron
    {
        template <bool bBiasInvolved, template<typename, typename> class TActFunc>
        struct TopoPicker_
        {
            using type = ComposeTopology<Sublayer<LinearSublayer, LinearLayer>,
                                         Sublayer<ActivateSublayer, TActFunc>,
                                         InConnect<LayerInput, LinearSublayer, LayerInput>,
                                         InternalConnect<LinearSublayer, LayerOutput, ActivateSublayer, LayerInput>,
                                         OutConnect<ActivateSublayer, LayerOutput, LayerOutput>>;

            template <typename TSublayerCont, typename TWeightShape, typename TBiasShape, typename... TParams>
            static auto ctor(const std::string& p_name, TSublayerCont&& sublayerCont, TWeightShape&& weightShape,
                             TBiasShape&& biasShape, TParams&&... params)
            {
                return std::forward<TSublayerCont>(sublayerCont)
                        .template Set<LinearSublayer>(p_name, std::forward<TWeightShape>(weightShape), std::forward<TBiasShape>(biasShape))
                        .template Set<ActivateSublayer>(p_name + "/act", std::forward<TParams>(params)...);
            }
        };
        
        template <template<typename, typename> class TActFunc>
        struct TopoPicker_<false, TActFunc>
        {
            using type = ComposeTopology<Sublayer<WeightParamSublayer, ParamSourceLayer>,
                                         Sublayer<DotSublayer, DotLayer>,
                                         Sublayer<ActivateSublayer, TActFunc>,
                                         InConnect<LayerInput, DotSublayer, LeftOperand>,
                                         InternalConnect<WeightParamSublayer, LayerOutput, DotSublayer, RightOperand>,
                                         InternalConnect<DotSublayer, LayerOutput, ActivateSublayer, LayerInput>,
                                         OutConnect<ActivateSublayer, LayerOutput, LayerOutput>>;

            template <typename TSublayerCont, typename TWeightShape, typename... TParams>
            static auto ctor(const std::string& p_name, TSublayerCont&& sublayerCont, TWeightShape&& weightShape, TParams&&... params)
            {
                return std::forward<TSublayerCont>(sublayerCont)
                        .template Set<WeightParamSublayer>(p_name + "/weight", std::forward<TWeightShape>(weightShape))
                        .template Set<DotSublayer>(p_name + "/dot")
                        .template Set<ActivateSublayer>(p_name + "/act", std::forward<TParams>(params)...);
            }
        };
        
        template <typename TInput, typename TPolicies>
        struct KernelConstructor_
        {
            using PlainPolicies = PlainPolicy<TPolicies>;
            using PolicySelectRes = PolicySelect<LayerStructurePolicy, PlainPolicies>;
            constexpr static bool biasInvolved = PolicySelectRes::BiasInvolved;
            
            template <typename UInput, typename UPolicies>
            using ActFunc = typename PolicySelectRes::template ActFunc<UInput, UPolicies>;
            
            static_assert(!LayerStructurePolicy::template IsDummyActFun<ActFunc>,
                          "Use PActFuncIs<...> to set activate function.");

            using TopoPickRes = TopoPicker_<biasInvolved, ActFunc>;
            using type = ComposeKernel<LayerPortSet<LayerInput>, LayerPortSet<LayerOutput>, TInput, TPolicies,
                                       typename TopoPickRes::type>;
            
            template <typename... TParams>
            static auto ctor(const std::string& layerName, TParams&&... params)
            {
                return TopoPickRes::ctor(layerName, type::CreateSublayers(), std::forward<TParams>(params)...);
            }
        };
    }
    
    template <typename TInput, typename TPolicies>
    class SingleLayerPerceptron : public NSSingleLayerPerceptron::KernelConstructor_<TInput, TPolicies>::type
    {
        using TKernelCtor = NSSingleLayerPerceptron::KernelConstructor_<TInput, TPolicies>;
        using TBase = typename TKernelCtor::type;

    public:
        template <typename... TParams>
        SingleLayerPerceptron(const std::string& p_name, TParams&&... params)
            : TBase(TKernelCtor::ctor(p_name, std::forward<TParams>(params)...))
        { }
    };
}
