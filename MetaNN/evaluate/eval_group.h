#pragma once

#include <MetaNN/evaluate/eval_item.h>
#include <list>
#include <memory>

namespace MetaNN
{
    template <typename TDevice>
    class BaseEvalGroup
    {
    public:
        using DeviceType = TDevice;
        virtual ~BaseEvalGroup() = default;
        
        virtual bool CanAdd(const BaseEvalItem<TDevice>&) = 0;
        virtual void Add(std::unique_ptr<BaseEvalItem<TDevice>>) = 0;
        virtual void Eval() = 0;
        virtual std::list<const void*> ResultPointers() const = 0;
    };
    
    template <typename TEvalItem>
    class TrivalEvalGroup : public BaseEvalGroup<typename TEvalItem::DeviceType>
    {
    public:
        using DeviceType = typename TEvalItem::DeviceType;
        virtual bool CanAdd(const BaseEvalItem<DeviceType>&) override final
        {
            return m_evalItem == nullptr;
        }
        
        virtual void Add(std::unique_ptr<BaseEvalItem<DeviceType>> item) override final
        {
            if (m_evalItem)
            {
                throw std::runtime_error("Cannot add to this group any more!");
            }
            BaseEvalItem<DeviceType>* to = item.release();
            auto aim = static_cast<TEvalItem*>(to);
            m_evalItem = std::unique_ptr<TEvalItem>(aim);
        }

        void Eval() override final
        {
            if (!m_evalItem)
                throw std::runtime_error("No eval item added now.");
            EvalInternalLogic(*m_evalItem);
        }
        
        virtual std::list<const void*> ResultPointers() const override final
        {
            if (!m_evalItem)
            {
                throw std::runtime_error("No eval item added now.");
            }
            return { m_evalItem->OutputPtr() };
        }

    protected:
        virtual void EvalInternalLogic(TEvalItem&) = 0;
    private:
        std::unique_ptr<TEvalItem> m_evalItem;
    };
}