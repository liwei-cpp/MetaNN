#pragma once
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <MetaNN/data_copy/data_copy.h>
namespace MetaNN
{
template <typename TElem, typename TPolicyCont, typename TFillers>
class ParamInitializer
{
public:
    using PolicyCont = TPolicyCont;
    
    ParamInitializer(TFillers&& filler)
        : m_filler(std::move(filler))
    {}
    
    template <typename TTag, typename TVal>
    auto SetFiller(TVal&& val) &&
    {
        auto newFiller = std::move(m_filler).template Set<TTag, TVal>(std::forward<TVal>(val));
        using newFillerType = RemConstRef<decltype(newFiller)>;
        return ParamInitializer<TElem, TPolicyCont, newFillerType>(std::move(newFiller));
    }
    
    template <typename TTag>
    auto& GetFiller()
    {
        return m_filler.template Get<TTag>();
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

template <typename TElem, typename...TPolicies>
auto MakeInitializer()
{
    using npType = FillerTags2NamedParams<TPolicies...>;
    using FilDictType = RemConstRef<decltype(npType::Create())>;
    return ParamInitializer<TElem, PolicyContainer<TPolicies...>, FilDictType>(npType::Create());
}
}