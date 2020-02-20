#pragma once
#include <string>
#include <stdexcept>
#include <typeindex>
#include <unordered_map>

namespace MetaNN
{
    namespace NSWeightBuffer
    {
        struct BaseCont
        {
            virtual ~BaseCont() = default;
        };
        
        template <typename T>
        struct Cont : BaseCont
        {
            std::unordered_map<std::string, T> data;
        };
    };
    
    class WeightBuffer
    {
    public:
        template <typename TType>
        NSWeightBuffer::Cont<TType>* GetCont() const
        {
            const std::type_index typeID = std::type_index(typeid(TType));
            auto it = m_params.find(typeID);
            if (it == m_params.end())
            {
                return nullptr;
            }
            
            return static_cast<NSWeightBuffer::Cont<TType>*>(it->second.get());
        }

        template <typename T>
        void Set(const std::string& name, T&& t)
        {
            using RawT = RemConstRef<T>;
            const std::type_index typeID = std::type_index(typeid(RawT));

            auto it = m_params.find(typeID);
            if (it == m_params.end())
            {
                m_params.emplace(typeID, std::make_unique<NSWeightBuffer::Cont<RawT>>());
                it = m_params.find(typeID);
            }
            NSWeightBuffer::Cont<RawT>* contPtr = static_cast<NSWeightBuffer::Cont<RawT>*>(it->second.get());
            if (contPtr->data.find(name) != contPtr->data.end())
            {
                throw std::runtime_error("Duplicate parameter with name: " + name);
            }
            
            contPtr->data.emplace(name, std::forward<T>(t));
        }

        template <typename T>
        const T* TryGet(const std::string& name) const
        {
            const std::type_index typeID = std::type_index(typeid(T));
            auto it = m_params.find(typeID);
            
            if (it == m_params.end())
            {
                return nullptr;
            }
            
            const auto* cont = static_cast<NSWeightBuffer::Cont<T>*>(it->second.get());
            
            auto it2 = cont->data.find(name);
            if (it2 == cont->data.end())
            {
                return nullptr;
            }
            return &(it2->second);
        }
        
        template <typename T>
        T* TryGet(const std::string& name)
        {
            const std::type_index typeID = std::type_index(typeid(T));
            auto it = m_params.find(typeID);
            
            if (it == m_params.end())
            {
                return nullptr;
            }
            
            auto* cont = static_cast<NSWeightBuffer::Cont<T>*>(it->second.get());
            
            auto it2 = cont->data.find(name);
            if (it2 == cont->data.end())
            {
                return nullptr;
            }
            return &(it2->second);
        }
        
        template <typename TType>
        bool IsExist(const std::string& name) const
        {
            NSWeightBuffer::Cont<TType>* cont = GetCont<TType>();
            if (!cont) return false;
            auto it = cont->data.find(name);
            return it != cont->data.end();
        }
        
        void Clear()
        {
            m_params.clear();
        }        
    private:
        std::unordered_map<std::type_index, std::unique_ptr<NSWeightBuffer::BaseCont>> m_params;
    };
}
