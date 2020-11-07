#pragma once

#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/facilities/_.h>
#include <algorithm>

namespace MetaNN
{
    namespace NSScalableTensor
    {
        template <size_t OriDim>
        auto ShapeInit(const Shape<OriDim>& ori)
        {
            Shape<OriDim + 1> res;
            if constexpr (OriDim != 0)
            {
                std::copy(std::begin(ori), std::end(ori), std::begin(res) + 1);
            }
            res[0] = 0;
            return res;
        }
        
        template <typename TInputHandle, typename TOutputHandle, size_t uDim>
        class EvalItem : public BaseEvalItem
        {
        public:
            using DeviceType = DeviceTypeFromHandle<TOutputHandle>;
            EvalItem(std::vector<TInputHandle> p_input, TOutputHandle p_output,
                     std::set<const void*> p_dependencies,
                     Shape<uDim> p_outputShape)
                : BaseEvalItem(TypeID<EvalItem>(), std::move(p_dependencies), p_output.DataPtr())
                , m_inputs(std::move(p_input))
                , m_output(std::move(p_output))
                , m_outputShape(std::move(p_outputShape))
            {}

            std::vector<TInputHandle> m_inputs;
            TOutputHandle m_output;
            Shape<uDim> m_outputShape;
        };
        
        template <typename TInputHandle, typename TOutputHandle, size_t uDim>
        class EvalGroup : public TrivialEvalGroup<EvalItem<TInputHandle, TOutputHandle, uDim>>
        {
            using EvalItemType = EvalItem<TInputHandle, TOutputHandle, uDim>;
        protected:
            virtual void EvalInternalLogic(EvalItemType& evalItem) final override
            {
                using ResType = typename TOutputHandle::DataType;
                using ElementType = typename ResType::ElementType;
                ResType res(evalItem.m_outputShape);

                static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>,
                              "Currently only CPU is supported");

                if (!evalItem.m_inputs.empty())
                {
                    auto lowerRes = LowerAccess(res);
                    ElementType* resMem = lowerRes.MutableRawMemory();

                    const size_t itemCount = evalItem.m_inputs[0].Data().Shape().Count();
                    for (size_t i = 0; i < evalItem.m_inputs.size(); ++i)
                    {
                        const auto& curItem = evalItem.m_inputs[i].Data();
                        assert(curItem.Shape() == evalItem.m_inputs[0].Data().Shape());

                        auto lowerItem = LowerAccess(curItem);
                        const ElementType* itemMem = lowerItem.RawMemory();
                        std::copy(itemMem, itemMem + itemCount, resMem);
                        resMem += itemCount;
                    }
                }

                evalItem.m_output.SetData(std::move(res));
            }
        };
    }
    
    template <typename TData>
    class ScalableTensor
    {
        static_assert(std::is_same_v<RemConstRef<TData>, TData>);
        
        using ElemCate = typename TData::CategoryTag;
        constexpr static size_t ElemShapeDim = ElemCate::DimNum;
    public:
        using CategoryTag = CategoryTags::Tensor<ElemShapeDim + 1>;
        using ElementType = typename TData::ElementType;
        using DeviceType = typename TData::DeviceType;


    public:
        explicit ScalableTensor() = default;
        
        template <typename... TShapeParameter,
                  std::enable_if_t<(std::is_convertible_v<TShapeParameter, size_t> && ...)>* = nullptr>
        explicit ScalableTensor(TShapeParameter... shapes)
            : m_shape(0, shapes...)
        {}
        
        explicit ScalableTensor(const MetaNN::Shape<ElemShapeDim>& itemShape)
        {
            for (size_t i = 0; i < ElemShapeDim; ++i)
            {
                m_shape[i + 1] = itemShape[i];
            }
            m_shape[0] = 0;
        }
        
        template <typename TIterator,
                  std::enable_if_t<IsIterator<TIterator>>* = nullptr>
        ScalableTensor(TIterator b, TIterator e)
            : m_shape(NSScalableTensor::ShapeInit(b->Shape()))
        {
            if (b == e)
            {
                throw std::runtime_error("Cannot initialize dynamic array with empty sequence.");
            }

            for (auto cur = b; cur != e; ++cur)
            {
                PushBack(*cur);
            }
        }

        const auto& Shape() const noexcept
        {
            return m_shape;
        }

        bool AvailableForWrite() const noexcept
        {
            return (!m_evalBuf.IsEvaluated()) && (m_buffer.use_count() == 1);
        }

        void PushBack(TData data)
        {
            if (m_buffer->empty())
            {
                m_shape = NSScalableTensor::ShapeInit(data.Shape());
            }
            
            if constexpr (ElemShapeDim != 0)
            {
                if (!std::equal(std::begin(m_shape) + 1, std::end(m_shape), std::begin(data.Shape())))
                {
                    throw std::runtime_error("Shape mismatch");
                }
            }
            m_buffer->push_back(std::move(data));
            ++m_shape[0];
        }

        void Reserve(size_t num)
        {
            assert(AvailableForWrite());
            m_buffer.reserve(num);
        }
    
        void Clear()
        {
            assert(AvailableForWrite());
            m_buffer.clear();
            m_shape = NSScalableTensor::ShapeInit(m_shape.Cardinal());
        }
    
        void Reverse()
        {
            assert(AvailableForWrite());
            if (!m_buffer) return;
            auto& cont = *m_buffer;
            std::reverse(cont.begin(), cont.end());
        }
    
        bool IsEmpty() const
        {
            return m_buffer->empty();
        }
    
        const auto& operator[] (size_t id) const
        {
            return (*m_buffer)[id];
        }
    
        bool operator== (const ScalableTensor& val) const
        {
            return m_buffer == val.m_buffer;
        }
        
        auto EvalRegister() const
        {
            if (!m_evalBuf.IsEvaluated())
            {
                auto outHandle = m_evalBuf.Handle();
                if (!EvalPlan::Inst().IsAlreadyRegisted(outHandle.DataPtr()))
                {
                    using TOpEvalHandle = std::decay_t<decltype(std::declval<TData>().EvalRegister())>;

                    std::vector<TOpEvalHandle> handleBuf;
                    std::set<const void*> depVec;
                    handleBuf.reserve(m_buffer->size());
                    for (size_t i = 0; i < m_buffer->size(); ++i)
                    {
                        handleBuf.push_back((*m_buffer)[i].EvalRegister());
                        depVec.insert(handleBuf.back().DataPtr());
                    }

                    using ItemType = NSScalableTensor::EvalItem<TOpEvalHandle, decltype(outHandle), ElemShapeDim + 1>;
                    using GroupType = NSScalableTensor::EvalGroup<TOpEvalHandle, decltype(outHandle), ElemShapeDim + 1>;
                    using DispatcherType = TrivialEvalItemDispatcher<GroupType>;

                    auto item = std::make_unique<ItemType>(std::move(handleBuf), std::move(outHandle), std::move(depVec), m_shape);
                    EvalPlan::Inst().Register<DispatcherType>(std::move(item));
                }
            }
            return m_evalBuf.ConstHandle();
        }

    private:
        MetaNN::Shape<ElemShapeDim + 1> m_shape;
        std::shared_ptr<std::vector<TData>> m_buffer = std::make_shared<std::vector<TData>>();
        
        using PrincipleType = PrincipalDataType<CategoryTag, ElementType, DeviceType>;
        EvalBuffer<PrincipleType> m_evalBuf;
    };
}