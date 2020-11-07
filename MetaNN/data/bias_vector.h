#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_buffer.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <MetaNN/facilities/_.h>
#include <cassert>
#include <cstring>
#include <type_traits>

namespace MetaNN
{
    namespace NSBiasVector
    {
        template <typename TInputHandle, typename TOutputHandle>
        class EvalItem : public BaseEvalItem
        {
        public:
            using DeviceType = DeviceTypeFromHandle<TOutputHandle>;

            EvalItem(TInputHandle inputHandle, TOutputHandle outputHandle,
                     size_t vecLen, size_t pos)
                : BaseEvalItem(TypeID<EvalItem>(),
                               { inputHandle.DataPtr() }, outputHandle.DataPtr())
                , m_inputHandle(std::move(inputHandle))
                , m_resHandle(std::move(outputHandle))
                , m_vecLen(vecLen)
                , m_pos(pos)
            {}

            TInputHandle  m_inputHandle;
            TOutputHandle m_resHandle;
            size_t m_vecLen;
            size_t m_pos;
        };

        template <typename TInputHandle, typename TOutputHandle>
        class EvalGroup : public TrivialEvalGroup<EvalItem<TInputHandle, TOutputHandle>>
        {
            using EvalItemType = EvalItem<TInputHandle, TOutputHandle>;
        protected:
            virtual void EvalInternalLogic(EvalItemType& evalItem) final override
            {
                using ResType = typename TOutputHandle::DataType;
                using ElementType = typename ResType::ElementType;

                static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>,
                              "Currently only CPU is supported.");

                ResType out(evalItem.m_vecLen);
                auto lowLayer = LowerAccess(out);
                auto mem = lowLayer.MutableRawMemory();
                
                memset(mem, 0, sizeof(ElementType) * evalItem.m_vecLen);
                mem[evalItem.m_pos] = evalItem.m_inputHandle.Data().Value();
                evalItem.m_resHandle.SetData(std::move(out));
            }
        };
    }

    template<typename TScalar>
    class BiasVector
    {
    public:
        using CategoryTag = CategoryTags::Tensor<1>;
        using ElementType = typename TScalar::ElementType;
        using DeviceType = typename TScalar::DeviceType;

    public:
        explicit BiasVector(size_t vecLen, size_t pos, TScalar scalar)
            : m_shape(MetaNN::Shape(vecLen))
            , m_pos(pos)
            , m_scalar(std::move(scalar))
        {
            assert(pos < vecLen);
        }

        const auto& Shape() const noexcept
        {
            return m_shape;
        }

        bool operator== (const BiasVector& val) const
        {
            return (m_shape == val.m_shape) &&
                   (m_pos == val.m_pos) &&
                   (m_scalar == val.m_scalar);
        }

        auto EvalRegister() const
        {
            if (!m_evalBuf.IsEvaluated())
            {
                auto outHandle = m_evalBuf.Handle();
        
                if (!EvalPlan::Inst().IsAlreadyRegisted(outHandle.DataPtr()))
                {
                    auto handle = m_scalar.EvalRegister();

                    using ItemType = NSBiasVector::EvalItem<decltype(handle), decltype(outHandle)>;
                    using GroupType = NSBiasVector::EvalGroup<decltype(handle), decltype(outHandle)>;
                    using DispatcherType = TrivialEvalItemDispatcher<GroupType>;

                    auto item = std::make_unique<ItemType>(std::move(handle), std::move(outHandle), m_shape[0], m_pos);
                    EvalPlan::Inst().Register<DispatcherType>(std::move(item));
                }
            }
            return m_evalBuf.ConstHandle();
        }
        
        size_t HotPos() const
        {
            return m_pos;
        }

        const auto& Scalar() const
        {
            return m_scalar;
        }

    private:
        MetaNN::Shape<1> m_shape;
        size_t m_pos;
        TScalar  m_scalar;
        EvalBuffer<Tensor<ElementType, DeviceType, 1>> m_evalBuf;
    };
}