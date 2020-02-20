#pragma once

#include <MetaNN/model/facilities/weight_buffer.h>
#include <MetaNN/data/_.h>
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
            using AimType = PrincipalDataType<TCategory, TElem, TDevice>;
            return m_weightBuffer.TryGet<AimType>(name);
        }
        
        template <typename TData>
        void Set(const std::string& name, const TData& data)
        {
            static_assert(std::is_same_v<typename TData::ElementType, TElem>);
            static_assert(std::is_same_v<typename TData::DeviceType, TDevice>);
            m_weightBuffer.Set(name, data);
        }
        
        void Clear()
        {
            m_weightBuffer.Clear();
        }
        
        template <typename TParamCate>
        bool IsParamExist(const std::string& name) const
        {
            using AimType = PrincipalDataType<TParamCate, TElem, TDevice>;
            return m_weightBuffer.IsExist<AimType>(name);
        }
        
    private:
        WeightBuffer m_weightBuffer;
    };
}