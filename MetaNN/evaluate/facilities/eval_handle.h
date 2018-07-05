#pragma once

#include <cassert>
#include <memory>
#include <stdexcept>

namespace MetaNN
{
template <typename TData>
class EvalHandle
{
    struct DataWithEvalInfo
    {
        TData m_data;
        bool m_eval = false;
    };
    
public:
    EvalHandle()
        : m_data(std::make_shared<DataWithEvalInfo>())
    {}

    bool IsEvaluated() const noexcept
    {
        return m_data->m_eval;
    }
    
    TData& MutableData()
    {
        if (IsEvaluated())
        {
            throw std::runtime_error("Data is already evaluated.");
        }
        return m_data->m_data;
    }
    
    void SetEval()
    {
        if (IsEvaluated())
        {
            throw std::runtime_error("Data is already evaluated.");
        }
        m_data->m_eval = true;
    }
    
    const TData& Data() const
    {
        if (!IsEvaluated())
        {
            throw std::runtime_error("Data is not evaluated.");
        }
        return m_data->m_data;
    }
    
    const void* DataPtr() const
    {
        return m_data.get();
    }

    template <typename...TParams>
    void Allocate(TParams&&... params) const
    {
        if (IsEvaluated())
        {
            throw std::runtime_error("Data is already evaluated.");
        }
        m_data->m_data = TData(std::forward<TParams>(params)...);
    }

private:
    std::shared_ptr<DataWithEvalInfo> m_data;
};

template <typename TData>
class ConstEvalHandle
{
public:
    ConstEvalHandle(TData data)
        : m_constData(std::move(data))
    {}
    
    const TData& Data() const
    {
        return m_constData;
    }
    
    const void* DataPtr() const
    {
        return &m_constData;
    }
    
private:
    TData m_constData;
};

template <typename TData>
class ConstEvalHandle<EvalHandle<TData>>
{
public:
    ConstEvalHandle(EvalHandle<TData> data)
        : m_constData(std::move(data))
    {}
    
    const TData& Data() const
    {
        return m_constData.Data();
    }
    
    const void* DataPtr() const
    {
        return m_constData.DataPtr();
    }
    
private:
    EvalHandle<TData> m_constData;
};

template <typename TData>
auto MakeConstEvalHandle(const TData& data)
{
    return ConstEvalHandle<TData>(data);
}

namespace NSEvalHandle
{
template <typename TData>
class DynamicHandleDataBase
{
public:
    virtual ~DynamicHandleDataBase() = default;
    virtual const TData& Data() const = 0;
    virtual const void* DataPtr() const = 0;
};

template <typename TData>
class DynamicHandleData;

template <typename TData>
class DynamicHandleData<ConstEvalHandle<TData>>
    : public DynamicHandleDataBase<TData>
{
public:
    DynamicHandleData(ConstEvalHandle<TData> data)
        : DynamicHandleDataBase<TData>()
        , m_data(std::move(data)) {}
        
    const TData& Data() const override
    {
        return m_data.Data();
    }
    
    const void* DataPtr() const override
    {
        return m_data.DataPtr();
    }

private:
    ConstEvalHandle<TData> m_data;
};

template <typename TData>
class DynamicHandleData<ConstEvalHandle<EvalHandle<TData>>>
    : public DynamicHandleDataBase<TData>
{
public:
    DynamicHandleData(ConstEvalHandle<EvalHandle<TData>> data)
        : DynamicHandleDataBase<TData>()
        , m_data(std::move(data)) {}
        
    const TData& Data() const override
    {
        return m_data.Data();
    }
    
    const void* DataPtr() const override
    {
        return m_data.DataPtr();
    }

private:
    ConstEvalHandle<EvalHandle<TData>> m_data;
};
}

template <typename TData>
class DynamicConstEvalHandle
{
    using TBaseData = NSEvalHandle::DynamicHandleDataBase<TData>;
public:
    template <typename TRealHandle>
    DynamicConstEvalHandle(TRealHandle data)
        : m_data(std::make_shared<NSEvalHandle::DynamicHandleData<TRealHandle>>(std::move(data)))
    {
        assert(m_data);
    }
    
    const TData& Data() const
    {
        return m_data->Data();
    }
    
    const void* DataPtr() const
    {
        return m_data->DataPtr();
    }
    
private:
    std::shared_ptr<TBaseData> m_data;
};
}