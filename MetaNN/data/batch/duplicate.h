#pragma once

namespace MetaNN
{
template <typename TData, typename TDataCate>
class DuplicateImp;

template <typename TData>
class Duplicate : public DuplicateImp<TData, DataCategory<TData>>
{
public:
    using ElementType = typename TData::ElementType;
    using DeviceType = typename TData::DeviceType;
    using DuplicateImp<TData, DataCategory<TData>>::DuplicateImp;
};

template <typename TData>
constexpr bool IsBatchMatrix<Duplicate<TData>> = IsMatrix<TData>;

template <typename TData>
constexpr bool IsBatchScalar<Duplicate<TData>> = IsScalar<TData>;

namespace NSDuplicate
{
template <typename TInHandle, typename TElem, typename TDevice, typename TCategory>
struct EvalUnit;

template <typename TInHandle, typename TElem>
struct EvalUnit<TInHandle, TElem, DeviceTags::CPU, CategoryTags::Matrix>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    EvalUnit(TInHandle oper, size_t batchNum,
             EvalHandle<Batch<TElem, DeviceTags::CPU, CategoryTags::Matrix>> evalOutput)
        : m_oper(std::move(oper))
        , m_batchNum(batchNum)
        , m_evalOutput(std::move(evalOutput)) { }

    void Eval() override
    {
        const auto& p_v1 = m_oper.Data();
        const size_t rowNum = p_v1.RowNum();
        const size_t colNum = p_v1.ColNum();

        m_evalOutput.Allocate(m_batchNum, rowNum, colNum);
        auto& res = m_evalOutput.MutableData();

        for (size_t i = 0; i < m_batchNum; ++i)
        {
            for (size_t j = 0; j < rowNum; ++j)
            {
                for (size_t k = 0; k < colNum; ++k)
                {
                    res.SetValue(i, j, k, p_v1(j, k));
                }
            }
        }
        m_evalOutput.SetEval();
    }

private:
    TInHandle m_oper;
    size_t m_batchNum;
    EvalHandle<Batch<TElem, DeviceTags::CPU, CategoryTags::Matrix>> m_evalOutput;
};

template <typename TInHandle, typename TElem>
struct EvalUnit<TInHandle, TElem, DeviceTags::CPU, CategoryTags::Scalar>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    EvalUnit(TInHandle oper, size_t batchNum,
             EvalHandle<Batch<TElem, DeviceTags::CPU, CategoryTags::Scalar>> evalOutput)
        : m_oper(std::move(oper))
        , m_batchNum(batchNum)
        , m_evalOutput(std::move(evalOutput)) { }

    void Eval() override
    {
        const auto& p_v1 = m_oper.Data();

        m_evalOutput.Allocate(m_batchNum);
        auto& res = m_evalOutput.MutableData();

        for (size_t i = 0; i < m_batchNum; ++i)
        {
            res.SetValue(i, p_v1.Value());
        }
        m_evalOutput.SetEval();
    }

private:
    TInHandle m_oper;
    size_t m_batchNum;
    EvalHandle<Batch<TElem, DeviceTags::CPU, CategoryTags::Scalar>> m_evalOutput;
};
}

template <typename TData>
class DuplicateImp<TData, CategoryTags::Matrix>
{
    static_assert(std::is_same<RemConstRef<TData>, TData>::value, "Unavailable data type");
    
public:
    using ElementType = typename TData::ElementType;
    using DeviceType = typename TData::DeviceType;

    DuplicateImp(TData data, size_t batch_num)
        : m_data(std::move(data))
        , m_batchNum(batch_num)
    {
        assert(m_batchNum != 0);
    }
        
public:
    size_t RowNum() const { return m_data.RowNum(); }
    size_t ColNum() const { return m_data.ColNum(); }
    
    size_t BatchNum() const { return m_batchNum; }
    
    const TData& Element() const
    {
        return m_data;
    }
    
    bool operator== (const Duplicate<TData>& val) const
    {
        const DuplicateImp<TData, CategoryTags::Matrix>& tmp = static_cast<const DuplicateImp<TData, CategoryTags::Matrix>&>(val);
        return (tmp.m_data == m_data) && (tmp.m_batchNum == m_batchNum);
    }

    template <typename TOtherType>
    bool operator== (const TOtherType&) const
    {
        return false;
    }

    template <typename TCompData>
    bool operator!= (const TCompData& val) const
    {
        return !(operator==(val));
    }
    
    auto EvalRegister() const
    {
        if (!m_evalBuf.IsEvaluated())
        {
            auto inHandle = m_data.EvalRegister();
            auto outHandle = m_evalBuf.Handle();
            
            using EvalUnit = NSDuplicate::EvalUnit<decltype(inHandle), 
                                                   ElementType, DeviceType, CategoryTags::Matrix>;
            using GroupType = TrivalEvalGroup<EvalUnit>;
            
            const void* dataPtr = outHandle.DataPtr();
            const void* depPtr = inHandle.DataPtr();
            EvalUnit unit(std::move(inHandle), m_batchNum, std::move(outHandle));
            EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, {depPtr});
        }
        return m_evalBuf.ConstHandle();
    }
    
protected:
    TData m_data;
    size_t m_batchNum;
    EvalBuffer<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalBuf;
};

template <typename TData>
class DuplicateImp<TData, CategoryTags::Scalar>
{
    static_assert(std::is_same<RemConstRef<TData>, TData>::value, "Unavailable data type");
    
public:
    using ElementType = typename TData::ElementType;
    using DeviceType = typename TData::DeviceType;

    DuplicateImp(TData data, size_t batch_num)
        : m_data(std::move(data))
        , m_batchNum(batch_num)
    {
        assert(m_batchNum != 0);
    }
        
public:
    size_t Size() const { return m_batchNum; }
    
    const TData& Element() const
    {
        return m_data;
    }
    
    bool operator== (const Duplicate<TData>& val) const
    {
        const DuplicateImp<TData, CategoryTags::Scalar>& tmp = static_cast<const DuplicateImp<TData, CategoryTags::Scalar>&>(val);
        return (tmp.m_data == m_data) && (tmp.m_batchNum == m_batchNum);
    }

    template <typename TOtherType>
    bool operator== (const TOtherType&) const
    {
        return false;
    }

    template <typename TCompData>
    bool operator!= (const TCompData& val) const
    {
        return !(operator==(val));
    }
    
    auto EvalRegister() const
    {
        if (!m_evalBuf.IsEvaluated())
        {
            auto inHandle = m_data.EvalRegister();
            auto outHandle = m_evalBuf.Handle();
            
            using EvalUnit = NSDuplicate::EvalUnit<decltype(inHandle), 
                                                   ElementType, DeviceType, CategoryTags::Scalar>;
            using GroupType = TrivalEvalGroup<EvalUnit>;
            
            const void* dataPtr = outHandle.DataPtr();
            const void* depPtr = inHandle.DataPtr();
            EvalUnit unit(std::move(inHandle), m_batchNum, std::move(outHandle));
            EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, {depPtr});
        }
        return m_evalBuf.ConstHandle();
    }
    
protected:
    TData m_data;
    size_t m_batchNum;
    EvalBuffer<Batch<ElementType, DeviceType, CategoryTags::Scalar>> m_evalBuf;
};

template<typename TData>
auto MakeDuplicate(size_t batchNum, TData&& data)
{
    using RawDataType = RemConstRef<TData>;
    return Duplicate<RawDataType>(std::forward<TData>(data), batchNum);
}

template<typename TData, typename...TParams>
auto MakeDuplicate(size_t batchNum, TParams&&... data)
{
    using RawDataType = RemConstRef<TData>;
    RawDataType tmp(std::forward<TParams>(data)...);
    return Duplicate<RawDataType>(std::move(tmp), batchNum);
}
}