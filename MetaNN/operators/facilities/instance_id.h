#pragma once
#include <atomic>

namespace MetaNN
{
    class InstanceID
    {
    public:
        InstanceID() = delete;
        static size_t Get()
        {
            return m_counter.fetch_add(1);
        }
    private:
        inline static std::atomic<size_t> m_counter = 0;
    };
}