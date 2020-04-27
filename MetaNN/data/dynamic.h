#pragma once

namespace MetaNN
{
    template <typename TElem, typename TDevice, typename TDataCate>
    class DynamicBase
    {
    protected:
        using EvalType = PrincipalDataType<TDataCate, TElem, TDevice>;

    public:
        virtual ~DynamicBase() = default;

        virtual const MetaNN::Shape<TDataCate::DimNum>& Shape() const = 0;

        virtual bool operator== (const DynamicBase& val) const = 0;

        virtual DynamicConstEvalHandle<EvalType> EvalRegister() const = 0;
    };
    
    template <typename TInternalData>
    class DynamicWrapper : public DynamicBase<typename TInternalData::ElementType,
                                              typename TInternalData::DeviceType,
                                              typename TInternalData::CategoryTag>
    {
        using TBase = DynamicBase<typename TInternalData::ElementType,
                                  typename TInternalData::DeviceType,
                                  typename TInternalData::CategoryTag>;
    public:
        DynamicWrapper(TInternalData data)
            : m_internal(std::move(data)) {}

        const MetaNN::Shape<TInternalData::CategoryTag::DimNum>&
        Shape() const override final
        {
            return m_internal.Shape();
        }
    
        bool operator== (const TBase& val) const override final
        {
            try
            {
                const DynamicWrapper& real = dynamic_cast<const DynamicWrapper&>(val);
                return m_internal == real.m_internal;
            }
            catch(std::bad_cast&)
            {
                return false;
            }
        }

        DynamicConstEvalHandle<typename TBase::EvalType> EvalRegister() const override final
        {
            return m_internal.EvalRegister();
        }
    
        const TInternalData& Internal() const
        {
            return m_internal;
        }

    private:
        TInternalData m_internal;
    };
    
    template <typename TElem, typename TDevice, typename TDataCate>
    class DynamicData
    {
        using InternalType = DynamicBase<TElem, TDevice, TDataCate>;

    public:
        using ElementType = TElem;
        using DeviceType = TDevice;
        using CategoryTag = TDataCate;

    public:
        DynamicData() = default;
    
        template <typename TOriData>
        DynamicData(std::shared_ptr<DynamicWrapper<TOriData>> data)
            : m_internal(std::move(data))
        { }
    
        const auto& Shape() const
        {
            if (!m_internal)
            {
                throw std::runtime_error("Invalid internal buffer");
            }
            return m_internal->Shape();
        }

        auto EvalRegister() const
        {
            if (!m_internal)
            {
                throw std::runtime_error("Invalid internal buffer");
            }
            return m_internal->EvalRegister();
        }

        bool operator== (const DynamicData& val) const
        {
            if ((!m_internal) && (!val.m_internal))
            {
                return true;
            }
            if ((!m_internal) || (!val.m_internal))
            {
                return false;
            }
            const InternalType& val1 = *m_internal;
            const InternalType& val2 = *(val.m_internal);
            return val1 == val2;
        }

        template <typename T>
        const T* TryCastTo() const
        {
            const auto* ptr = m_internal.get();
            auto ptrCast = dynamic_cast<const DynamicWrapper<T>*>(ptr);

            return (ptrCast ? &(ptrCast->Internal()) : nullptr);
        }
    
        bool IsEmpty() const
        {
            return m_internal == nullptr;
        }
    private:
        std::shared_ptr<InternalType> m_internal;
    };

    template <typename TData>
    constexpr bool IsDynamic_ = false;

    template <typename TElem, typename TDevice, typename TCate>
    constexpr bool IsDynamic_<DynamicData<TElem, TDevice, TCate>> = true;
    
    template <typename TData>
    constexpr bool IsDynamic = IsDynamic_<RemConstRef<TData>>;

    template <typename TData>
    auto MakeDynamic(TData&& data)
    {
        if constexpr (IsDynamic<TData>)
        {
            return std::forward<TData>(data);
        }
        else
        {
            using RawData = RemConstRef<TData>;
            using TDeriveData = DynamicWrapper<RawData>;
            auto internal = std::make_shared<TDeriveData>(std::forward<TData>(data));
            return DynamicData<typename RawData::ElementType,
                               typename RawData::DeviceType,
                               DataCategory<RawData>>(std::move(internal));
        }
    }
}