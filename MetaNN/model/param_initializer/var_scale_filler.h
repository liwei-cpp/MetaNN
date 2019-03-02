#pragma once

#include <random>
#include <stdexcept>
#include <type_traits>
#include <MetaNN/model/param_initializer/facilities/fill_with_spec_dist.h>
#include <MetaNN/policies/change_policy.h>

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
    TypePolicyObj(PNormVarScale,    VarScaleFillerPolicy, Distribute, Norm);
    TypePolicyObj(PUniformVarScale, VarScaleFillerPolicy, Distribute, Uniform);
    TypePolicyObj(PVarScaleFanIn,   VarScaleFillerPolicy, ScaleMode,  FanIn);
    TypePolicyObj(PVarScaleFanOut,  VarScaleFillerPolicy, ScaleMode,  FanOut);
    TypePolicyObj(PVarScaleFanAvg,  VarScaleFillerPolicy, ScaleMode,  FanAvg);
#include <MetaNN/policies/policy_macro_end.h>

    namespace NSVarScaleFiller
    {
        template <typename TElem, typename TDevice>
        auto GetFanInFanOut(const Scalar<TElem, TDevice>&)
        {
            return {1, 1};
        }
        
        template <typename TElem, typename TDevice>
        auto GetFanInFanOut(const Matrix<TElem, TDevice>& mat)
        {
            return {mat.Shape().RowNum(), mat.Shape.ColNum()};
        }
        
        template <typename TElem, typename TDevice>
        auto GetFanInFanOut(const ThreeDArray<TElem, TDevice>& arr)
        {
            return {1, arr.Shape().PageNum()};
        }
        
        template <typename TElem, typename TDevice>
        auto GetFanInFanOut(const ThreeDArraySequence<TElem, TDevice>& data)
        {
            return {data.Shape().Length(), data.Shape().PageNum()};
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
        void Fill(TData& data) const
        {
            auto [fanIn, fanOut] = NSVarScaleFiller::GetFanInFanOut(data);
            using ScaleMode = typename PolicySelect<VarScaleFillerPolicy, TPolicyCont>::ScaleMode;        
            double fan_factor = 0;
            if constexpr (std::is_same<ScaleMode, VarScaleFillerPolicy::ScaleModeTypeCate::FanIn>::value)
            {
                fan_factor = fanIn;
            }
            else if constexpr (std::is_same<ScaleMode, VarScaleFillerPolicy::ScaleModeTypeCate::FanOut>::value)
            {
                fan_factor = fanOut;
            }
            else if constexpr (std::is_same<ScaleMode, VarScaleFillerPolicy::ScaleModeTypeCate::FanAvg>::value)
            {
                fan_factor = (fanIn + fanOut) / 2;
            }
            else
            {
                static_assert(DependencyFalse<ScaleMode>);
            }
        
            using DistType = typename PolicySelect<VarScaleFillerPolicy, TPolicyCont>::Distribute;                                            
            using ElementType = typename TData::ElementType;
            if constexpr (std::is_same<DistType, VarScaleFillerPolicy::DistributeTypeCate::Uniform>::value)
            {
                double limit = sqrt(3.0 * m_factor / fan_factor);
                std::uniform_real_distribution<ElementType> dist(-limit, limit);
                NSInitializer::FillWithDist(data, dist, m_engine);
            }
            else if constexpr (std::is_same<DistType, VarScaleFillerPolicy::DistributeTypeCate::Norm>::value)
            {
                double stddev = sqrt(m_factor / fan_factor);
                std::normal_distribution<ElementType> dist(0, stddev);
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