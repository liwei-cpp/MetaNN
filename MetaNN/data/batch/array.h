#pragma once

#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/batch/batch.h>
#include <MetaNN/evaluate/facilities/eval_buffer.h>
#include <MetaNN/evaluate/facilities/eval_group.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <cassert>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace MetaNN
{
template <typename TData, typename TDataCate>
class ArrayImp;

template <typename TData>
class Array : public ArrayImp<TData, DataCategory<TData>>
{
public:
    using ElementType = typename TData::ElementType;
    using DeviceType = typename TData::DeviceType;
    using ArrayImp<TData, DataCategory<TData>>::ArrayImp;
};

template <typename TData>
constexpr bool IsBatchMatrix<Array<TData>> = IsMatrix<TData>;

template <typename TData>
constexpr bool IsBatchScalar<Array<TData>> = IsScalar<TData>;

namespace NSArray
{
template <typename TInputElem, typename TElem, typename TDevice, typename TCategory>
struct EvalUnit;

template <typename TInputElem, typename TElem>
struct EvalUnit<TInputElem, TElem, DeviceTags::CPU, CategoryTags::Matrix>
    : public BaseEvalUnit<DeviceTags::CPU>
{
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
    
    EvalUnit(std::vector<TInputElem> p_input,
             EvalHandle<Batch<TElem, DeviceTags::CPU, CategoryTags::Matrix>> p_output)
        : m_inputs(std::move(p_input))
        , m_output(std::move(p_output))
        {}

    void Eval()
    {
        if (m_inputs.empty())
        {
            m_output.Allocate(0, 0, 0);
        }
        else
        {
            size_t tbn = m_inputs.size();
            size_t trn = m_inputs[0].Data().RowNum();
            size_t tcn = m_inputs[0].Data().ColNum();
            m_output.Allocate(tbn, trn, tcn);
            auto& res = m_output.MutableData();
            
            for (size_t bn = 0; bn < tbn; ++bn)
            {
                const auto& input = m_inputs[bn].Data();
                
                for (size_t i = 0; i < trn; ++i)
                {
                    for (size_t j = 0; j < tcn; ++j)
                    {
                        res.SetValue(bn, i, j, input(i, j));
                    }
                }
            }
        }
        m_output.SetEval();
    }
    
private:
    std::vector<TInputElem> m_inputs;
    EvalHandle<Batch<TElem, DeviceTags::CPU, CategoryTags::Matrix>> m_output;
};

template <typename TInputElem, typename TElem>
struct EvalUnit<TInputElem, TElem, DeviceTags::CPU, CategoryTags::Scalar>
    : public BaseEvalUnit<DeviceTags::CPU>
{
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
    
    EvalUnit(std::vector<TInputElem> p_input,
             EvalHandle<Batch<TElem, DeviceTags::CPU, CategoryTags::Scalar>> p_output)
        : m_inputs(std::move(p_input))
        , m_output(std::move(p_output))
        {}

    void Eval()
    {
        if (m_inputs.empty())
        {
            m_output.Allocate(0);
        }
        else
        {
            size_t tbn = m_inputs.size();
            m_output.Allocate(tbn);
            auto& res = m_output.MutableData();
            
            for (size_t bn = 0; bn < tbn; ++bn)
            {
                res.SetValue(bn, m_inputs[bn].Data().Value());
            }
        }
        m_output.SetEval();
    }
    
private:
    std::vector<TInputElem> m_inputs;
    EvalHandle<Batch<TElem, DeviceTags::CPU, CategoryTags::Scalar>> m_output;
};
}

template <typename TData>
class ArrayImp<TData, CategoryTags::Matrix>
{
public:
    using ElementType = typename TData::ElementType;
    using DeviceType = typename TData::DeviceType;

    ArrayImp(size_t rowNum = 0, size_t colNum = 0)
        : m_rowNum(rowNum)
        , m_colNum(colNum)
        , m_buffer(new std::vector<TData>()){}
        
    template <typename TIterator, std::enable_if_t<IsIterator<TIterator>>* = nullptr>
    ArrayImp(TIterator b, TIterator e)
        : m_rowNum(0)
        , m_colNum(0)
        , m_buffer(new std::vector<TData>(b, e))
    {
        const auto& buffer = *m_buffer;
        if (!buffer.empty())
        {
            m_rowNum = buffer[0].RowNum();
            m_colNum = buffer[0].ColNum();
            
            for (size_t i = 1; i < buffer.size(); ++i)
            {
                if ((buffer[i].RowNum() != m_rowNum) ||
                    (buffer[i].ColNum() != m_colNum))
                {
                    throw std::runtime_error("Dimension mismatch");
                }
            }
        }
    }
    
public:
    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }
    
    size_t BatchNum() const
    {
        return m_buffer->size();
    }
    
    size_t size() const { return m_buffer->size(); }
    

    void push_back(TData mat)
    {
        assert(AvailableForWrite());
        if ((mat.RowNum() != m_rowNum) || (mat.ColNum() != m_colNum))
        {
            throw std::runtime_error("Dimension mismatch");
        }
        m_buffer->emplace_back(std::move(mat));
    }
    
    template <typename...TArgs>
    void emplace_back(TArgs&&... args)
    {
        assert(AvailableForWrite());
        TData tmp(std::forward<TArgs>(args)...);
        if ((tmp.RowNum() != m_rowNum) || (tmp.ColNum() != m_colNum))
        {
            throw std::runtime_error("Dimension mismatch");
        }
        m_buffer.emplace_back(std::move(tmp));
    }
    
    void reserve(size_t num)
    {
        assert(AvailableForWrite());
        m_buffer.reserve(num);
    }
    
    void clear()
    {
        assert(AvailableForWrite());
        m_buffer.clear();
    }
    
    bool empty() const
    {
        return m_buffer->empty();
    }
    
    const auto& operator[] (size_t id) const
    {
        return (*m_buffer)[id];
    }
    
    auto& operator[] (size_t id)
    {
        return (*m_buffer)[id];
    }
    
    auto begin() { return m_buffer->begin(); }
    auto begin() const { return m_buffer->begin(); }
    auto end() { return m_buffer->end(); }
    auto end() const { return m_buffer->end(); }
    
    bool operator== (const Array<TData>& val) const
    {
        const ArrayImp<TData, CategoryTags::Matrix>& tmp = static_cast<const ArrayImp<TData, CategoryTags::Matrix>&>(val);
        return m_buffer == tmp.m_buffer;
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
            using TOpEvalHandle = std::decay_t<decltype(std::declval<TData>().EvalRegister())>;
            std::vector<TOpEvalHandle> handleBuf;
            std::vector<const void*> depVec;
            handleBuf.reserve(this->size());
            depVec.reserve(this->size());
            for (size_t i = 0; i < this->size(); ++i)
            {
                handleBuf.push_back((*this)[i].EvalRegister());
                depVec.push_back(handleBuf.back().DataPtr());
            }
            
            auto outHandle = m_evalBuf.Handle();
            
            using EvalUnit = NSArray::EvalUnit<TOpEvalHandle, ElementType, DeviceType, CategoryTags::Matrix>;
            using GroupType = TrivalEvalGroup<EvalUnit>;
            
            const void* dataPtr = outHandle.DataPtr();
            EvalUnit unit(std::move(handleBuf), std::move(outHandle));
            EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, std::move(depVec));
        }
        return m_evalBuf.ConstHandle();
    }
    
    bool AvailableForWrite() const
    {
        return (!m_evalBuf.IsEvaluated())&&(m_buffer.use_count() == 1);
    }
    
protected:
    size_t m_rowNum;
    size_t m_colNum;
    std::shared_ptr<std::vector<TData>> m_buffer;
    EvalBuffer<Batch<ElementType, DeviceType, CategoryTags::Matrix>> m_evalBuf;
};

template <typename TData>
class ArrayImp<TData, CategoryTags::Scalar>
{
public:
    using ElementType = typename TData::ElementType;
    using DeviceType = typename TData::DeviceType;

    ArrayImp(size_t rowNum = 0, size_t colNum = 0)
        : m_buffer(new std::vector<TData>()){}
        
    template <typename TIterator, std::enable_if_t<IsIterator<TIterator>>* = nullptr>
    ArrayImp(TIterator b, TIterator e)
        : m_buffer(new std::vector<TData>(b, e))
    { }
    
public:
    size_t BatchNum() const
    {
        return m_buffer->size();
    }
    
    size_t size() const { return m_buffer->size(); }
    

    void push_back(TData mat)
    {
        assert(AvailableForWrite());
        m_buffer->emplace_back(std::move(mat));
    }
    
    template <typename...TArgs>
    void emplace_back(TArgs&&... args)
    {
        assert(AvailableForWrite());
        TData tmp(std::forward<TArgs>(args)...);
        m_buffer.emplace_back(std::move(tmp));
    }
    
    void reserve(size_t num)
    {
        assert(AvailableForWrite());
        m_buffer.reserve(num);
    }
    
    void clear()
    {
        assert(AvailableForWrite());
        m_buffer.clear();
    }
    
    bool empty() const
    {
        return m_buffer->empty();
    }
    
    const auto& operator[] (size_t id) const
    {
        return (*m_buffer)[id];
    }
    
    auto& operator[] (size_t id)
    {
        return (*m_buffer)[id];
    }
    
    auto begin() { return m_buffer->begin(); }
    auto begin() const { return m_buffer->begin(); }
    auto end() { return m_buffer->end(); }
    auto end() const { return m_buffer->end(); }
    
    bool operator== (const Array<TData>& val) const
    {
        const ArrayImp<TData, CategoryTags::Scalar>& tmp = static_cast<const ArrayImp<TData, CategoryTags::Scalar>&>(val);
        return m_buffer == tmp.m_buffer;
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
            using TOpEvalHandle = std::decay_t<decltype(std::declval<TData>().EvalRegister())>;
            std::vector<TOpEvalHandle> handleBuf;
            std::vector<const void*> depVec;
            handleBuf.reserve(this->size());
            depVec.reserve(this->size());
            for (size_t i = 0; i < this->size(); ++i)
            {
                handleBuf.push_back((*this)[i].EvalRegister());
                depVec.push_back(handleBuf.back().DataPtr());
            }
            
            auto outHandle = m_evalBuf.Handle();
            
            using EvalUnit = NSArray::EvalUnit<TOpEvalHandle, ElementType, DeviceType, CategoryTags::Scalar>;
            using GroupType = TrivalEvalGroup<EvalUnit>;
            
            const void* dataPtr = outHandle.DataPtr();
            EvalUnit unit(std::move(handleBuf), std::move(outHandle));
            EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, std::move(depVec));
        }
        return m_evalBuf.ConstHandle();
    }
    
    bool AvailableForWrite() const
    {
        return (!m_evalBuf.IsEvaluated())&&(m_buffer.use_count() == 1);
    }
    
protected:
    std::shared_ptr<std::vector<TData>> m_buffer;
    EvalBuffer<Batch<ElementType, DeviceType, CategoryTags::Scalar>> m_evalBuf;
};

template <typename TIterator>
auto MakeArray(TIterator beg, TIterator end)
{
    using TData = typename std::iterator_traits<TIterator>::value_type;
    using RawData = RemConstRef<TData>;
    
    return Array<RawData>(beg, end);
}
}