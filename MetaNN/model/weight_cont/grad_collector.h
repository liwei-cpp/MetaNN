#pragma once
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
        switch (grad.Shape().BatchNum())
        {
        case 0:
            throw std::runtime_error("Empty grad.");
        case 1:
            return MakeDynamic(grad[0]);
        default:
            return MakeDynamic(Collapse(grad, weight.Shape()));
        }
    }
    
private:
    WeightType weight;
    DynamicBatch<GradItemType> grad;
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
    void Collect(std::string_view name, const TWeight& weight, TGrad&& grad)
    {
        if constexpr (IsScalar<TWeight>)
        {
            auto it = GetOrCreateEntry(name, m_scalarGrad, m_scalarGradMap, weight);
            it->Push(std::forward<TGrad>(grad));
        }
        else if constexpr (IsMatrix<TWeight>)
        {
            auto it = GetOrCreateEntry(name, m_matrixGrad, m_matrixGradMap, weight);
            it->Push(std::forward<TGrad>(grad));
        }
        else if constexpr (IsThreeDArray<TWeight>)
        {
            auto it = GetOrCreateEntry(name, m_3dArrayGrad, m_3dArrayGradMap, weight);
            it->Push(std::forward<TGrad>(grad));
        }
        else
        {
            static_assert(DependencyFalse<TWeight>);
        }
    }

    void Clear()
    {
        m_scalarGrad.clear();
        m_scalarGradMap.clear();
        m_matrixGrad.clear();
        m_matrixGradMap.clear();
        m_3dArrayGrad.clear();
        m_3dArrayGradMap.clear();
    }

    template <typename TCategory>
    const auto& GetContainer() const
    {
        if constexpr (std::is_same_v<TCategory, CategoryTags::Scalar>)
        {
            return m_scalarGrad;
        }
        else if constexpr (std::is_same_v<TCategory, CategoryTags::Matrix>)
        {
            return m_matrixGrad;
        }
        else if constexpr (std::is_same_v<TCategory, CategoryTags::ThreeDArray>)
        {
            return m_3dArrayGrad;
        }
        else
        {
            static_assert(DependencyFalse<TCategory>);
        }
    }
    
    template <typename TCategory>
    const auto& GetGradInfo(std::string_view p_paramName)
    {
        if constexpr (std::is_same_v<TCategory, CategoryTags::Scalar>)
        {
            return GetEntry(p_paramName, m_scalarGradMap);
        }
        else if constexpr (std::is_same_v<TCategory, CategoryTags::Matrix>)
        {
            return GetEntry(p_paramName, m_matrixGradMap);
        }
        else if constexpr (std::is_same_v<TCategory, CategoryTags::ThreeDArray>)
        {
            return GetEntry(p_paramName, m_3dArrayGradMap);
        }
        else
        {
            static_assert(DependencyFalse<TCategory>);
        }
    }

private:
    template <typename TCategory>
    using GradContainer = std::list<GradInfo<TElement, TDevice, TCategory>>;
    
    template <typename TCategory>
    using GradContIter = typename GradContainer<TCategory>::iterator;
    
    GradContainer<CategoryTags::Scalar> m_scalarGrad;
    std::unordered_map<std::string_view, GradContIter<CategoryTags::Scalar>> m_scalarGradMap;
    
    GradContainer<CategoryTags::Matrix> m_matrixGrad;
    std::unordered_map<std::string_view, GradContIter<CategoryTags::Matrix>> m_matrixGradMap;
    
    GradContainer<CategoryTags::ThreeDArray> m_3dArrayGrad;
    std::unordered_map<std::string_view, GradContIter<CategoryTags::ThreeDArray>> m_3dArrayGradMap;
};
}
