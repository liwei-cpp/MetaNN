#pragma once
#include <atomic>

namespace MetaNN
{
    namespace NSTypeID
    {
        inline size_t GenTypeID()
        {
            static std::atomic<size_t> m_counter = 0;
            return m_counter.fetch_add(1);
        }
    }

    template <typename T>
    size_t TypeID()
    {
        const static size_t id = NSTypeID::GenTypeID();
        return id;
    }
}