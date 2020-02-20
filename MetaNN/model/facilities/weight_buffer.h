#pragma once
#include <MetaNN/facilities/_.h>
#include <string>
#include <stdexcept>
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
            const size_t typeID = TypeID<TType>();
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
            auto* contPtr = GetCont<RawT>();
            if (!contPtr)
            {
                const size_t typeID = TypeID<RawT>();
                m_params.emplace(typeID, std::make_unique<NSWeightBuffer::Cont<RawT>>());
                contPtr = GetCont<RawT>();
            }

            if (contPtr->data.find(name) != contPtr->data.end())
            {
                throw std::runtime_error("Duplicate parameter with name: " + name);
            }
            
            contPtr->data.emplace(name, std::forward<T>(t));
        }

        template <typename T>
        const T* TryGet(const std::string& name) const
        {
            const auto* cont = GetCont<T>();
            if (!cont) return nullptr;

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
            auto* cont = GetCont<T>();
            if (!cont) return nullptr;
            
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
        std::unordered_map<size_t, std::unique_ptr<NSWeightBuffer::BaseCont>> m_params;
    };
}
