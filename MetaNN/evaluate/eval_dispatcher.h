#pragma once
#include <list>
#include <memory>
#include <MetaNN/evaluate/eval_group.h>

namespace MetaNN
{
    class BaseEvalItemDispatcher
    {
    public:
        BaseEvalItemDispatcher(size_t evalItemID)
            : m_evalItemID(evalItemID)
        {}
        virtual ~BaseEvalItemDispatcher() = default;
        virtual void Add(std::unique_ptr<BaseEvalItem>) = 0;

        virtual size_t MaxEvalGroupSize() const = 0;
        
        virtual std::unique_ptr<BaseEvalGroup> PickNextGroup() = 0;
    protected:
        const size_t m_evalItemID;
    };
    
    template <typename TEvalGroup>
    class TrivialEvalItemDispatcher final : public BaseEvalItemDispatcher
    {
    public:
        TrivialEvalItemDispatcher(size_t evalItemID)
            : BaseEvalItemDispatcher(evalItemID) {}

        virtual void Add(std::unique_ptr<BaseEvalItem> item) final override
        {
            assert(BaseEvalItemDispatcher::m_evalItemID == item->ID());
            m_evalItems.push_back(std::move(item));
        }
        
        virtual size_t MaxEvalGroupSize() const final override
        {
            return m_evalItems.empty() ? 0 : 1;
        }
        
        virtual std::unique_ptr<BaseEvalGroup> PickNextGroup() final override
        {
            if (m_evalItems.empty()) return nullptr;
            std::unique_ptr<BaseEvalItem> curItem = std::move(m_evalItems.front());
            m_evalItems.pop_front();
            auto res = std::make_unique<TEvalGroup>();
            res->Add(std::move(curItem));
            return res;
        }

    private:
        std::list<std::unique_ptr<BaseEvalItem>> m_evalItems;
    };
}