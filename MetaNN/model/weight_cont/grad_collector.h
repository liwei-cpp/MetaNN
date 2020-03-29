#pragma once
#include <MetaNN/model/facilities/weight_buffer.h>
#include <MetaNN/operation/math/reduce_sum.h>
#include <list>
#include <stdexcept>
#include <unordered_map>

namespace MetaNN
{
template <typename TElement, typename TDevice, typename TCategory>
struct GradInfo
{
    using GradItemType = DynamicData<TElement, TDevice, TCategory>;
    using WeightType = PrincipalDataType<TCategory, TElement, TDevice>;
    
    GradInfo(WeightType p_weight)
        : weight(std::move(p_weight))
        , grad(weight.Shape())
    {}
    
    template <typename TGrad>
    void Push(TGrad&& p_grad)
    {
        auto tmp = MakeDynamic(std::forward<TGrad>(p_grad));
        static_assert(std::is_same_v<decltype(tmp), GradItemType>);
        if (tmp.Shape() != weight.Shape())
        {
            throw std::runtime_error("Shape of Weight/Grad mismatch.");
        }
        grad.PushBack(std::move(tmp));
    }
    
    const auto& Weight() const
    {
        return weight;
    }
    
    auto Grad() const
    {
        switch (grad.Shape()[0])
        {
        case 0:
            throw std::runtime_error("Empty grad.");
        case 1:
            return MakeDynamic(grad[0]);
        default:
            return MakeDynamic(ReduceSum<PolicyContainer<PModifyDimNumIs<1>>>(grad));
        }
    }
    
private:
    WeightType weight;
    ScalableTensor<GradItemType> grad;
};

template <typename TElement, typename TDevice>
class GradCollector
{
private:
    template <typename TCont, typename TMap, typename TWeight>
    auto GetOrCreateEntry(std::string_view name, TCont& p_cont, TMap& p_map, const TWeight& p_weight)
    {
        auto it = p_map.find(name);
        if (it != p_map.end()) return it->second;
        
        using TGradInfo = typename TCont::value_type;
        p_cont.push_front(TGradInfo(p_weight));
        p_map.insert({name, p_cont.begin()});
        return p_cont.begin();
    }
    
    template <typename TMap>
    const auto& GetEntry(std::string_view name, TMap& p_map)
    {
        auto it = p_map.find(name);
        if (it == p_map.end())
        {
            throw std::runtime_error(std::string("Cannot find the parameter with name: ") + std::string(name));
        }
        return *(it->second);
    }
    
public:
    template<typename TWeight, typename TGrad>
    void Collect(const std::string& name, const TWeight& weight, TGrad&& grad)
    {
        using TGradInfo = GradInfo<TElement, TDevice, typename TWeight::CategoryTag>;
        
        auto* ptr = m_weightBuffer.TryGet<TGradInfo>(name);
        if (!ptr)
        {
            m_weightBuffer.Set(name, TGradInfo(weight));
            ptr = m_weightBuffer.TryGet<TGradInfo>(name);
        }
        ptr->Push(std::forward<TGrad>(grad));
    }

    void Clear()
    {
        m_weightBuffer.Clear();
    }

    template <typename TCategory>
    const auto& GetContainer() const
    {
        using TGradInfo = GradInfo<TElement, TDevice, TCategory>;

        auto* cont = m_weightBuffer.GetCont<TGradInfo>();
        if (!cont)
        {
            throw std::runtime_error("Grad container not exist");
        }
        return cont->data;
    }
    
private:
    WeightBuffer m_weightBuffer;
};
}
