#pragma once

#include <random>
#include <stdexcept>
#include <type_traits>
#include <MetaNN/model_rel/param_initializer/facilities/fill_with_spec_dist.h>

namespace MetaNN
{
template <typename TPolicyCont = PolicyContainer<>>
class UniformFiller
{
    using TRandomEngine = typename PolicySelect<InitPolicy, TPolicyCont>::RandEngine;
public:
    UniformFiller(double min, double max,
                  unsigned seed = std::random_device{}())
        : m_engine(seed)
        , m_min(min)
        , m_max(max)
    {
        if (min >= max)
        {
            throw std::runtime_error("min is larger or equal to max");
        }
    }
    
    template <typename TData>
    void Fill(TData& data, size_t /*fanin*/, size_t /*fanout*/)
    {
        using ElementType = typename TData::ElementType;
        using DistType = std::conditional_t<std::is_integral<ElementType>::value,
                                            std::uniform_int_distribution<ElementType>,
                                            std::uniform_real_distribution<ElementType>>;
        DistType dist(m_min, m_max);
        NSInitializer::FillWithDist(data, dist, m_engine);
    }
    
private:
    TRandomEngine m_engine;
    double m_min;
    double m_max;
};
}