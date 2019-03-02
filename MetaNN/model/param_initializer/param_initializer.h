#pragma once
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <MetaNN/data_copy/data_copy.h>
namespace MetaNN
{
    template <typename TElem, typename TFillers>
    class ParamInitializer
    {
    public:
        ParamInitializer(TFillers&& filler)
            : m_filler(std::move(filler))
        {}
        
        template <typename TKey>
        const auto& GetFiller() const
        {
            return m_filler.template Get<TKey>();
        }
        
        template <typename TElem2, typename TDevice2>
        void SetParam(const std::string& name, const Scalar<TElem2, TDevice2>& param)
        {
            if (m_scalarParams.find(name) != m_scalarParams.end())
            {
                throw std::runtime_error("Duplicate parameter scalar: " + name);
            }
        
            if constexpr (std::is_same<TElem2, TElem>::value &&
                          std::is_same<TDevice2, DeviceTags::CPU>::value)
            {
                m_scalarParams.insert({name, param});
            }
            else
            {
                Scalar<TElem, DeviceTags::CPU> data(param.Shape());
                DataCopy(param, data);
                m_scalarParams.insert({name, std::move(data)});
            }
        }
    
        template <typename TElem2, typename TDevice2>
        void SetParam(const std::string& name, const Matrix<TElem2, TDevice2>& param)
        {
            if (m_matrixParams.find(name) != m_matrixParams.end())
            {
                throw std::runtime_error("Duplicate parameter matrix: " + name);
            }
        
            if constexpr (std::is_same<TElem2, TElem>::value &&
                          std::is_same<TDevice2, DeviceTags::CPU>::value)
            {
                m_matrixParams.insert({name, param});
            }
            else
            {
                Matrix<TElem, DeviceTags::CPU> mat(param.Shape());
                DataCopy(param, mat);
                m_matrixParams.insert({name, std::move(mat)});
            }
        }
    
        template <typename TElem2, typename TDevice2>
        void SetParam(const std::string& name, const ThreeDArray<TElem2, TDevice2>& param)
        {
            if (m_3dArrayParams.find(name) != m_3dArrayParams.end())
            {
                throw std::runtime_error("Duplicate parameter 3d-array: " + name);
            }
        
            if constexpr (std::is_same<TElem2, TElem>::value &&
                          std::is_same<TDevice2, DeviceTags::CPU>::value)
            {
                m_3dArrayParams.insert({name, param});
            }
            else
            {
                ThreeDArray<TElem, DeviceTags::CPU> data(param.Shape());
                DataCopy(param, data);
                m_3dArrayParams.insert({name, std::move(data)});
            }
        }
    
        template <typename TElem2, typename TDevice2>
        void GetParam(const std::string& name, Scalar<TElem2, TDevice2>& res) const
        {
            auto it = m_scalarParams.find(name);
            if (it == m_scalarParams.end())
            {
                throw std::runtime_error("Scalar parameter not exist: " + name);
            }
            const auto& oriData = it->second;
            if (oriData.Shape() != res.Shape())
            {
                throw std::runtime_error("Scalars shape mismatch.");
            }
            DataCopy(oriData, res);
        }
    
        template <typename TElem2, typename TDevice2>
        void GetParam(const std::string& name, Matrix<TElem2, TDevice2>& res) const
        {
            auto it = m_matrixParams.find(name);
            if (it == m_matrixParams.end())
            {
                throw std::runtime_error("Matrix parameter not exist: " + name);
            }
            const auto& oriData = it->second;
            if (oriData.Shape() != res.Shape())
            {
                throw std::runtime_error("Matrices shape mismatch.");
            }
            DataCopy(oriData, res);
        }
    
        template <typename TElem2, typename TDevice2>
        void GetParam(const std::string& name, ThreeDArray<TElem2, TDevice2>& res) const
        {
            auto it = m_3dArrayParams.find(name);
            if (it == m_3dArrayParams.end())
            {
                throw std::runtime_error("3d array parameter not exist: " + name);
            }
            const auto& oriData = it->second;
            if (oriData.Shape() != res.Shape())
            {
                throw std::runtime_error("3d arraies shape mismatch.");
            }
            DataCopy(oriData, res);
        }
    
        template <typename TParamCate>
        bool IsParamExist(const std::string& name) const
        {
            if constexpr (std::is_same_v<TParamCate, CategoryTags::Scalar>)
            {
                auto it = m_scalarParams.find(name);
                return it != m_scalarParams.end();
            }
            else if constexpr (std::is_same_v<TParamCate, CategoryTags::Matrix>)
            {
                auto it = m_matrixParams.find(name);
                return it != m_matrixParams.end();
            }
            else if constexpr (std::is_same_v<TParamCate, CategoryTags::ThreeDArray>)
            {
                auto it = m_3dArrayParams.find(name);
                return it != m_3dArrayParams.end();
            }
        }
    
    private:
        TFillers m_filler;
        std::unordered_map<std::string, Scalar<TElem, DeviceTags::CPU>> m_scalarParams;
        std::unordered_map<std::string, Matrix<TElem, DeviceTags::CPU>> m_matrixParams;
        std::unordered_map<std::string, ThreeDArray<TElem, DeviceTags::CPU>> m_3dArrayParams;
    };
    
    namespace NSMakeInitializer
    {
        template <typename TKey, typename TFiller>
        struct Initializer
        {
            using KeyType = TKey;
            TFiller m_filler;
        };
        
        template <typename TCont>
        auto CreateFillerCont(TCont&& cont)
        {
            return std::forward<TCont>(cont);
        }
        
        template <typename TCont, typename TCur, typename... TRemain>
        auto CreateFillerCont(TCont&& cont, TCur&& cur, TRemain&&... remain)
        {
            using TKey = typename RemConstRef<TCur>::KeyType;
            auto newCont = std::forward<TCont>(cont).template Set<TKey>(cur.m_filler);
            return CreateFillerCont(std::move(newCont), std::forward<TRemain>(remain)...);
        }
    }

    template <typename TInitKey, typename TFiller>
    auto InitializerKV(TFiller&& filler)
    {
        return NSMakeInitializer::Initializer<TInitKey, RemConstRef<TFiller>>{std::forward<TFiller>(filler)};
    }

    template <typename TElem, typename... TInitializers>
    inline auto MakeInitializer(TInitializers&&... fillers)
    {
        using FillContType = VarTypeDict<typename RemConstRef<TInitializers>::KeyType ...>;
        auto fillCont = NSMakeInitializer::CreateFillerCont(FillContType::Create(), std::forward<TInitializers>(fillers)...);
        
        return ParamInitializer<TElem, RemConstRef<decltype(fillCont)>>(std::move(fillCont));
    }
}