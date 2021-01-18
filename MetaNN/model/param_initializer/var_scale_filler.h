#pragma once

#include <random>
#include <stdexcept>
#include <type_traits>
#include <MetaNN/model/param_initializer/facilities/fill_with_spec_dist.h>
#include <MetaNN/policies/_.h>

namespace MetaNN
{
    struct VarScaleFillerPolicy
    {
        using MajorClass = VarScaleFillerPolicy;

        struct DistributeTypeCate
        {
            struct Uniform;
            struct Norm;
        };
        using Distribute = DistributeTypeCate::Uniform;
    
        struct ScaleModeTypeCate
        {
            struct FanIn;
            struct FanOut;
            struct FanAvg;
        };
        using ScaleMode = ScaleModeTypeCate::FanAvg;
    };
#include <MetaNN/policies/policy_macro_begin.h>
    EnumTypePolicyObj(PNormVarScale,    VarScaleFillerPolicy, Distribute, Norm);
    EnumTypePolicyObj(PUniformVarScale, VarScaleFillerPolicy, Distribute, Uniform);
    EnumTypePolicyObj(PVarScaleFanIn,   VarScaleFillerPolicy, ScaleMode,  FanIn);
    EnumTypePolicyObj(PVarScaleFanOut,  VarScaleFillerPolicy, ScaleMode,  FanOut);
    EnumTypePolicyObj(PVarScaleFanAvg,  VarScaleFillerPolicy, ScaleMode,  FanAvg);
#include <MetaNN/policies/policy_macro_end.h>

    namespace NSVarScaleFiller
    {
        template <typename TElem, typename TDevice, size_t uDim>
        auto GetFanInFanOut(const Tensor<TElem, TDevice, uDim>& t)
        {
            if constexpr (uDim == 1)
            {
                return std::tuple{1, 1};    
            }
            else if constexpr (uDim == 2)
            {
                return std::tuple{t.Shape()[0], t.Shape()[1]};
            }
            else if (uDim > 2)
            {
                const size_t count = t.Shape().Count();
                return std::tuple{count / t.Shape()[uDim - 1], count / t.Shape()[uDim-2]};
            }
            else
            {
                static_assert(DependencyFalse<Tensor<TElem, TDevice, uDim>>, "Dimension is 0.");
            }
        }
    }
    
    template <typename TPolicyCont = PolicyContainer<>>
    class VarScaleFiller
    {
        using TRandomEngine = typename PolicySelect<InitPolicy, TPolicyCont>::RandEngine;
    public:
        VarScaleFiller(double factor = 1,
                       unsigned seed = std::random_device{}())
            : m_factor(factor)
            , m_engine(seed)
        {}
    
        template <typename TData>
        void Fill(TData& data)
        {
            auto [fanIn, fanOut] = NSVarScaleFiller::GetFanInFanOut(data);
            using ScaleMode = typename PolicySelect<VarScaleFillerPolicy, TPolicyCont>::ScaleMode;        
            double fan_factor = 0;
            if constexpr (std::is_same<ScaleMode, VarScaleFillerPolicy::ScaleModeTypeCate::FanIn>::value)
            {
                fan_factor = static_cast<double>(fanIn);
            }
            else if constexpr (std::is_same<ScaleMode, VarScaleFillerPolicy::ScaleModeTypeCate::FanOut>::value)
            {
                fan_factor = fanOut;
            }
            else if constexpr (std::is_same<ScaleMode, VarScaleFillerPolicy::ScaleModeTypeCate::FanAvg>::value)
            {
                fan_factor = (fanIn + fanOut) / 2.;
            }
            else
            {
                static_assert(DependencyFalse<ScaleMode>);
            }
        
            using DistType = typename PolicySelect<VarScaleFillerPolicy, TPolicyCont>::Distribute;                                            
            using ElementType = typename TData::ElementType;
            if constexpr (std::is_same<DistType, VarScaleFillerPolicy::DistributeTypeCate::Uniform>::value)
            {
                ElementType limit = static_cast<ElementType>(sqrt(3.0 * m_factor / fan_factor));
                std::uniform_real_distribution<ElementType> dist(-limit, limit);
                NSInitializer::FillWithDist(data, dist, m_engine);
            }
            else if constexpr (std::is_same<DistType, VarScaleFillerPolicy::DistributeTypeCate::Norm>::value)
            {
                double stddev = sqrt(m_factor / fan_factor);
                std::normal_distribution<ElementType> dist(0, static_cast<ElementType>(stddev));
                NSInitializer::FillWithDist(data, dist, m_engine);
            }
            else
            {
                static_assert(DependencyFalse<DistType>);
            }
        }

    private:
        double m_factor; 
        TRandomEngine m_engine;
    };

    namespace NSVarScaleFiller
    {
        template <typename TPolicyCont>
        struct MSRAFillerPolicy_
        {
            using type1 = ChangePolicy<PNormVarScale, TPolicyCont>;
            using type = ChangePolicy<PVarScaleFanIn, type1>;
        };

        template <typename TPolicyCont>
        using MSRAFillerPolicy = typename MSRAFillerPolicy_<TPolicyCont>::type;
    }

    // MSRA Filler, use Norm Dist and FanIn Mode
    template <typename TPolicyCont = PolicyContainer<>>
    class MSRAFiller : public VarScaleFiller<NSVarScaleFiller::MSRAFillerPolicy<TPolicyCont>>
    {
        using BaseType = VarScaleFiller<NSVarScaleFiller::MSRAFillerPolicy<TPolicyCont>>;
    public:
        MSRAFiller(unsigned seed = std::random_device{}())
            : BaseType(2, seed) {}
    };

    // Xavier Filler, use FanAvg Mode
    template <typename TPolicyCont = PolicyContainer<>>
    class XavierFiller : public VarScaleFiller<ChangePolicy<PVarScaleFanAvg, TPolicyCont>>
    {
        using BaseType = VarScaleFiller<ChangePolicy<PVarScaleFanAvg, TPolicyCont>>;
    public:
        XavierFiller(unsigned seed = std::random_device{}())
            : BaseType(1, seed) {}
    };
}