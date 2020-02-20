#pragma once
#include <MetaNN/data_copy/data_copy.h>
#include <MetaNN/model/facilities/weight_buffer.h>
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
        auto& GetFiller()
        {
            return m_filler.template Get<TKey>();
        }
        
        template <typename TElem2, typename TDevice2, size_t uDim>
        void SetParam(const std::string& name, const Tensor<TElem2, TDevice2, uDim>& param)
        {
            using AimType = Tensor<TElem, DeviceTags::CPU, uDim>;
            if constexpr (std::is_same_v<AimType, Tensor<TElem2, TDevice2, uDim>>)
            {
                m_weightBuffer.Set(name, param);
            }
            else
            {
                AimType data(param.Shape());
                DataCopy(param, data);
                m_weightBuffer.Set(name, std::move(data));
            }
        }
    
        template <typename TElem2, typename TDevice2, size_t uDim>
        void GetParam(const std::string& name, Tensor<TElem2, TDevice2, uDim>& res) const
        {
            using AimType = Tensor<TElem, DeviceTags::CPU, uDim>;
            auto ptr = m_weightBuffer.TryGet<AimType>(name);
            if (!ptr)
            {
                throw std::runtime_error("Parameter not exist.");
            }
            if (ptr->Shape() != res.Shape())
            {
                throw std::runtime_error("Shape mismatch.");
            }
            DataCopy(*ptr, res);
        }
    
        template <typename TParamCate>
        bool IsParamExist(const std::string& name) const
        {
            using AimType = Tensor<TElem, DeviceTags::CPU, TParamCate::DimNum>;
            return m_weightBuffer.IsExist<AimType>(name);
        }
    
    private:
        TFillers m_filler;
        WeightBuffer m_weightBuffer;
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