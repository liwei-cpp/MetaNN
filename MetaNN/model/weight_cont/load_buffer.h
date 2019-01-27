#pragma once

#include <MetaNN/data/data.h>
#include <unordered_map>
#include <string>
#include <stdexcept>

namespace MetaNN
{
    template <typename TElem, typename TDevice>
    class LoadBuffer
    {
    public:
        LoadBuffer() = default;
        
        template <typename TCategory>
        const auto* TryGet(const std::string& name) const
        {
            if constexpr(std::is_same_v<TCategory, CategoryTags::Scalar>)
            {
                auto it = m_scalarParam.find(name);
                if (it == m_scalarParam.end())
                    return static_cast<Scalar<TElem, TDevice>*>(nullptr);
                return &(it->second);
            }
            else if constexpr(std::is_same_v<TCategory, CategoryTags::Matrix>)
            {
                auto it = m_matrixParam.find(name);
                if (it == m_matrixParam.end())
                    return static_cast<Matrix<TElem, TDevice>*>(nullptr);
                return &(it->second);
            }
            else if constexpr(std::is_same_v<TCategory, CategoryTags::ThreeDArray>)
            {
                auto it = m_3dArrayParam.find(name);
                if (it == m_3dArrayParam.end())
                    return static_cast<ThreeDArray<TElem, TDevice>*>(nullptr);
                return &(it->second);
            }
            else
            {
                static_assert(DependencyFalse<TCategory>);
                return nullptr;
            }
        }
        
        template <typename TData>
        void Set(const std::string& name, const TData& data)
        {
            if constexpr (std::is_same_v<TData, Scalar<TElem, TDevice>>)
            {
                auto tmp = TryGet<CategoryTags::Scalar>(name);
                if (tmp)
                {
                    throw std::runtime_error("Weight set duplicate.");
                }
                m_scalarParam.insert({name, data});
            }
            else if constexpr (std::is_same_v<TData, Matrix<TElem, TDevice>>)
            {
                auto tmp = TryGet<CategoryTags::Matrix>(name);
                if (tmp)
                {
                    throw std::runtime_error("Weight set duplicate.");
                }
                m_matrixParam.insert({name, data});
            }
            else if constexpr (std::is_same_v<TData, ThreeDArray<TElem, TDevice>>)
            {
                auto tmp = TryGet<CategoryTags::ThreeDArray>(name);
                if (tmp)
                {
                    throw std::runtime_error("Weight set duplicate.");
                }
                m_3dArrayParam.insert({name, data});
            }
            else
            {
                static_assert(DependencyFalse<TData>);
            }
        }
        
        void Clear()
        {
            m_scalarParam.clear();
            m_matrixParam.clear();
            m_3dArrayParam.clear();
        }
        
        template <typename TParamCate>
        bool IsParamExist(const std::string& name) const
        {
            if constexpr (std::is_same_v<TParamCate, CategoryTags::Scalar>)
            {
                auto it = m_scalarParam.find(name);
                return it != m_scalarParam.end();
            }
            else if constexpr (std::is_same_v<TParamCate, CategoryTags::Matrix>)
            {
                auto it = m_matrixParam.find(name);
                return it != m_matrixParam.end();
            }
            else if constexpr (std::is_same_v<TParamCate, CategoryTags::ThreeDArray>)
            {
                auto it = m_3dArrayParam.find(name);
                return it != m_3dArrayParam.end();
            }
        }
        
    private:
        std::unordered_map<std::string, Scalar<TElem, TDevice>> m_scalarParam;
        std::unordered_map<std::string, Matrix<TElem, TDevice>> m_matrixParam;
        std::unordered_map<std::string, ThreeDArray<TElem, TDevice>> m_3dArrayParam;
    };
}