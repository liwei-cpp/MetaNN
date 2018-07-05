#pragma once
#include <map>
#include <string>
#include <stdexcept>

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
    void SetMatrix(const std::string& name, const Matrix<TElem2, TDevice2>& param)
    {
        if (m_params.find(name) != m_params.end())
        {
            throw std::runtime_error("Duplicate parameter matrix: " + name);
        }
        
        if constexpr (std::is_same<TElem2, TElem>::value &&
                      std::is_same<TDevice2, DeviceTags::CPU>::value)
        {
            m_params.insert({name, param});
        }
        else
        {
            Matrix<TElem, DeviceTags::CPU> mat(param.RowNum(), param.ColNum());
            DataCopy(param, mat);
            m_params.insert({name, std::move(mat)});
        }
    }
    
    template <typename TElem2, typename TDevice2>
    void GetMatrix(const std::string& name, Matrix<TElem2, TDevice2>& res) const
    {
        auto it = m_params.find(name);
        if (it == m_params.end())
        {
            throw std::runtime_error("Parameter not exist: " + name);
        }
        const auto& oriMat = it->second;
        if ((oriMat.RowNum() != res.RowNum()) || (oriMat.ColNum() != res.ColNum()))
        {
            throw std::runtime_error("Matrices with different dimensions.");
        }
        
        DataCopy(oriMat, res);
    }
    
    bool IsMatrixExist(const std::string& name) const
    {
        auto it = m_params.find(name);
        return it != m_params.end();
    }
    
private:
    TFillers m_filler;
    std::map<std::string, Matrix<TElem, DeviceTags::CPU>> m_params;
};

template <typename TElem, typename...TPolicies>
auto MakeInitializer()
{
    using npType = FillerTags2NamedParams<TPolicies...>;
    using FilDictType = RemConstRef<decltype(npType::Create())>;
    return ParamInitializer<TElem, PolicyContainer<TPolicies...>, FilDictType>(npType::Create());
}
}