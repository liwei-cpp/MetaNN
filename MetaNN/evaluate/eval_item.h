#pragma once
#include <set>

namespace MetaNN
{
    class BaseEvalItem
    {
    public:
        BaseEvalItem(size_t evalItemID,
                     std::set<const void*>&& p_inputs, const void* p_output)
            : m_id(evalItemID)
            , m_inputPtrs(std::move(p_inputs))
            , m_outputPtr(p_output)
        {}

        virtual ~BaseEvalItem() = default;
        
        size_t ID() const { return m_id; }
        const std::set<const void*>& InputPtrs() const { return m_inputPtrs; }
        const void* OutputPtr() const { return m_outputPtr; }

    private:
        const size_t m_id;
        const std::set<const void*> m_inputPtrs;
        const void* m_outputPtr;
    };
}