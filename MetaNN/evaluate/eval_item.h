#pragma once

#include <typeindex>
#include <vector>

namespace MetaNN
{
    template <typename TDevice>
    class BaseEvalItem
    {
    public:
        using DeviceType = TDevice;
        BaseEvalItem(std::type_index evalItemID,
                     std::vector<const void*>&& p_inputs, const void* p_output)
            : m_id(evalItemID)
            , m_inputPtrs(std::move(p_inputs))
            , m_outputPtr(p_output)
        {}

        virtual ~BaseEvalItem() = default;
        
        std::type_index ID() const { return m_id; }
        const std::vector<const void*>& InputPtrs() const { return m_inputPtrs; }
        const void* OutputPtr() const { return m_outputPtr; }

    private:
        const std::type_index m_id;
        std::vector<const void*> m_inputPtrs;
        const void* m_outputPtr;
    };
}