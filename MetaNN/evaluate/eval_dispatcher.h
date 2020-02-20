#pragma once
#include <list>
#include <memory>
#include <MetaNN/evaluate/eval_group.h>

namespace MetaNN
{
    template <typename TDevice>
    class BaseEvalItemDispatcher
    {
    public:
        using DeviceType = TDevice;
        BaseEvalItemDispatcher(size_t evalItemID)
            : m_evalItemID(evalItemID)
        {}
        virtual ~BaseEvalItemDispatcher() = default;
        virtual void Add(std::unique_ptr<BaseEvalItem<DeviceType>>) = 0;

        virtual size_t MaxEvalGroupSize() const = 0;
        
        virtual std::unique_ptr<BaseEvalGroup<DeviceType>> PickNextGroup() = 0;
    protected:
        const size_t m_evalItemID;
    };
    
    template <typename TEvalGroup>
    class TrivalEvalItemDispatcher final
        : public BaseEvalItemDispatcher<typename TEvalGroup::DeviceType>
    {
        using TBase = BaseEvalItemDispatcher<typename TEvalGroup::DeviceType>;

    public:
        TrivalEvalItemDispatcher(size_t evalItemID)
            : TBase(evalItemID) {}

        virtual void Add(std::unique_ptr<BaseEvalItem<typename TBase::DeviceType>> item) final override
        {
            assert(TBase::m_evalItemID == item->ID());
            m_evalItems.push_back(std::move(item));
        }
        
        virtual size_t MaxEvalGroupSize() const final override
        {
            return m_evalItems.empty() ? 0 : 1;
        }
        
        virtual std::unique_ptr<BaseEvalGroup<typename TBase::DeviceType>> PickNextGroup() final override
        {
            if (m_evalItems.empty()) return nullptr;
            std::unique_ptr<BaseEvalItem<typename TBase::DeviceType>> curItem = std::move(m_evalItems.front());
            m_evalItems.pop_front();
            auto res = std::make_unique<TEvalGroup>();
            res->Add(std::move(curItem));
            return res;
        }

    private:
        std::list<std::unique_ptr<BaseEvalItem<typename TBase::DeviceType>>> m_evalItems;
    };
}