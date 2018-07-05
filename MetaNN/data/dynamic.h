#pragma once

#include <MetaNN/data/scalar.h>
#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/evaluate/facilities/eval_buffer.h>
#include <memory>

namespace MetaNN
{
template <typename TElem, typename TDevice, typename TDataCate>
class DynamicCategory;

template <typename TElem, typename TDevice>
class DynamicCategory<TElem, TDevice, CategoryTags::Matrix>
{
public:
    using ElementType = TElem;
    using DeviceType = TDevice;
    using EvalType = PrincipalDataType<CategoryTags::Matrix, ElementType, DeviceType>;

public:
    template <typename TBase>
    DynamicCategory(const TBase& base)
        : m_rowNum(base.RowNum())
        , m_colNum(base.ColNum()) {}

    virtual ~DynamicCategory() = default;

    virtual bool operator== (const DynamicCategory& val) const = 0;
    virtual bool operator!= (const DynamicCategory& val) const = 0;

    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }

    virtual DynamicConstEvalHandle<EvalType> EvalRegister() const = 0;

private:
    size_t m_rowNum;
    size_t m_colNum;
};

template <typename TElem, typename TDevice>
class DynamicCategory<TElem, TDevice, CategoryTags::BatchMatrix>
{
public:
    using ElementType = TElem;
    using DeviceType = TDevice;
    using EvalType = PrincipalDataType<CategoryTags::BatchMatrix, ElementType, DeviceType>;

public:
    template <typename TBase>
    DynamicCategory(const TBase& base)
        : m_rowNum(base.RowNum())
        , m_colNum(base.ColNum())
        , m_batchNum(base.BatchNum()){}

    virtual ~DynamicCategory() = default;

    virtual bool operator== (const DynamicCategory& val) const = 0;
    virtual bool operator!= (const DynamicCategory& val) const = 0;

    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }
    size_t BatchNum() const { return m_batchNum; }

    virtual DynamicConstEvalHandle<EvalType> EvalRegister() const = 0;

private:
    size_t m_rowNum;
    size_t m_colNum;
    size_t m_batchNum;
};

template <typename TBaseData>
class DynamicWrapper : public DynamicCategory<typename TBaseData::ElementType,
                                              typename TBaseData::DeviceType,
                                              DataCategory<TBaseData>>
{
    using TBase = DynamicCategory<typename TBaseData::ElementType,
                                  typename TBaseData::DeviceType,
                                  DataCategory<TBaseData>>;
public:
    DynamicWrapper(TBaseData data)
        : TBase(data)
        , m_baseData(std::move(data)) {}

    bool operator== (const TBase& val) const override
    {
        try
        {
            const DynamicWrapper& real = dynamic_cast<const DynamicWrapper&>(val);
            return m_baseData == real.m_baseData;
        }
        catch(std::bad_cast&)
        {
            return false;
        }
    }

    bool operator!= (const TBase& val) const override
    {
        return !(operator==(val));
    }

    DynamicConstEvalHandle<typename TBase::EvalType> EvalRegister() const override
    {
        return m_baseData.EvalRegister();
    }

    const TBaseData& BaseData() const
    {
        return m_baseData;
    }

private:
    TBaseData m_baseData;
};

template <typename TElem, typename TDevice, typename TDataCate>
class DynamicData;

template <typename TElem, typename TDevice>
class DynamicData<TElem, TDevice, CategoryTags::Matrix>
{
    using BaseData = DynamicCategory<TElem, TDevice, CategoryTags::Matrix>;
public:
    using ElementType = TElem;
    using DeviceType = TDevice;
    using ResHandleType = decltype(std::declval<BaseData>().EvalRegister());

    DynamicData() = default;
    
    template <typename TOriData>
    DynamicData(std::shared_ptr<DynamicWrapper<TOriData>> data)
    {
        m_baseData = std::move(data);
    }

    size_t RowNum() const { return m_baseData->RowNum(); }
    size_t ColNum() const { return m_baseData->ColNum(); }

    auto EvalRegister() const
    {
        return m_baseData->EvalRegister();
    }

    bool operator== (const DynamicData& val) const
    {
        if ((!m_baseData) && (!val.m_baseData))
        {
            return true;
        }
        if ((!m_baseData) || (!val.m_baseData))
        {
            return false;
        }
        BaseData& val1 = *m_baseData;
        BaseData& val2 = *(val.m_baseData);
        return val1 == val2;
    }

    template <typename TOtherType>
    bool operator== (const TOtherType& val) const
    {
        return false;
    }

    template <typename TOtherType>
    bool operator!= (const TOtherType& val) const
    {
        return !(operator==(val));
    }

    template <typename T>
    const T* TypeCast() const
    {
        const BaseData* ptr = m_baseData.get();
        auto ptrCast = dynamic_cast<const DynamicWrapper<T>*>(ptr);

        return (ptrCast ? &(ptrCast->BaseData()) : nullptr);
    }
    
    bool IsEmpty() const
    {
        return m_baseData == nullptr;
    }
private:
    std::shared_ptr<BaseData> m_baseData;
};

template <typename TElem, typename TDevice>
class DynamicData<TElem, TDevice, CategoryTags::BatchMatrix>
{
    using BaseData = DynamicCategory<TElem, TDevice, CategoryTags::BatchMatrix>;

public:
    using ElementType = TElem;
    using DeviceType = TDevice;
    using ResHandleType = decltype(std::declval<BaseData>().EvalRegister());

    DynamicData() = default;
    
    template <typename TOriData>
    DynamicData(std::shared_ptr<DynamicWrapper<TOriData>> data)
    {
        m_baseData = std::move(data);
    }

    bool operator== (const DynamicData& val) const
    {
        if ((!m_baseData) && (!val.m_baseData))
        {
            return true;
        }
        if ((!m_baseData) || (!val.m_baseData))
        {
            return false;
        }
        BaseData& val1 = *m_baseData;
        BaseData& val2 = *(val.m_baseData);
        return val1 == val2;
    }

    template <typename TOtherType>
    bool operator== (const TOtherType& val) const
    {
        return false;
    }

    template <typename TOtherType>
    bool operator!= (const TOtherType& val) const
    {
        return !(operator==(val));
    }

    size_t RowNum() const { return m_baseData->RowNum(); }
    size_t ColNum() const { return m_baseData->ColNum(); }
    size_t BatchNum() const { return m_baseData->BatchNum(); }

    auto EvalRegister() const
    {
        return m_baseData->EvalRegister();
    }

    template <typename T>
    const T* TypeCast() const
    {
        const BaseData* ptr = m_baseData.get();
        auto ptrCast = dynamic_cast<const DynamicWrapper<T>*>(ptr);

        return (ptrCast ? &(ptrCast->BaseData()) : nullptr);
    }
    
    bool IsEmpty() const
    {
        return m_baseData == nullptr;
    }
private:
    std::shared_ptr<BaseData> m_baseData;
};

template <typename TData>
constexpr bool IsDynamic = false;

template <typename TElem, typename TDevice, typename TCate>
constexpr bool IsDynamic<DynamicData<TElem, TDevice, TCate>> = true;

template <typename TElem, typename TDevice, typename TCate>
constexpr bool IsDynamic<DynamicData<TElem, TDevice, TCate>&> = true;

template <typename TElem, typename TDevice, typename TCate>
constexpr bool IsDynamic<DynamicData<TElem, TDevice, TCate>&&> = true;

template <typename TElem, typename TDevice, typename TCate>
constexpr bool IsDynamic<const DynamicData<TElem, TDevice, TCate>&> = true;

template <typename TElem, typename TDevice, typename TCate>
constexpr bool IsDynamic<const DynamicData<TElem, TDevice, TCate>&&> = true;

template <typename TData>
auto MakeDynamic(TData&& data)
{
    if constexpr (IsDynamic<TData>)
    {
        return std::forward<TData>(data);
    }
    else
    {
        using rawData = RemConstRef<TData>;
        using TDeriveData = DynamicWrapper<rawData>;
        auto baseData = std::make_shared<TDeriveData>(std::forward<TData>(data));
        return DynamicData<typename rawData::ElementType,
                           typename rawData::DeviceType,
                           DataCategory<rawData>>(std::move(baseData));
    }
}

template <typename TElem, typename TDevice>
constexpr bool IsMatrix<DynamicData<TElem, TDevice, CategoryTags::Matrix>> = true;

template <typename TElem, typename TDevice>
constexpr bool IsBatchMatrix<DynamicData<TElem, TDevice, CategoryTags::BatchMatrix>> = true;
}