#pragma once
#include <MetaNN/data/facilities/tags.h>
#include <cassert>
#include <array>
#include <stdexcept>

namespace MetaNN
{
    namespace DimConst
    {
        constexpr size_t Keep = 0;
        constexpr size_t Extend = (size_t)-1;
    }
    
    namespace NSShape
    {
        template <size_t uDimNum, typename TCurIndexType, typename... TRemainIndexType>
        size_t IndexToOffset(const std::array<size_t, uDimNum>& dims, size_t& gap, TCurIndexType curIdx, TRemainIndexType... remIdx)
        {
            constexpr size_t indexPos = uDimNum - sizeof...(TRemainIndexType) - 1;
            if (dims[indexPos] == DimConst::Extend)
                throw std::runtime_error("Invalid dimension value.");
            if (static_cast<size_t>(curIdx) >= dims[indexPos])
                throw std::runtime_error("Invalid dimension index.");

            if constexpr (sizeof...(TRemainIndexType) == 0)
            {
                gap = 1;
                return static_cast<size_t>(curIdx);
            }
            else
            {
                size_t curGap = 0;
                size_t res = IndexToOffset(dims, curGap, remIdx...);
                gap = curGap * dims[indexPos + 1];
                res += static_cast<size_t>(curIdx) * gap;
                return res;
            }
        }
    }

    template <size_t uDimNum>
    class Shape
    {
        static_assert(uDimNum > 0);
    public:
        constexpr static size_t DimNum = uDimNum;

        explicit Shape() = default;
        
        explicit Shape(std::array<size_t, uDimNum> dims)
            : m_dims(std::move(dims)) {}
        
        bool operator == (const Shape& val) const
        {
            return m_dims == val.m_dims;
        }
        
        template <size_t vDimNum>
        bool operator == (const Shape<vDimNum>&) const
        {
            return false;
        }
        
        size_t Count() const
        {
            size_t res = 1;
            for (size_t i = 0; i < uDimNum; ++i)
            {
                if (m_dims[i] == DimConst::Extend)
                    return DimConst::Extend;
                if (m_dims[i] == DimConst::Keep)
                    return DimConst::Keep;
                res *= m_dims[i];
            }
            return res;
        }
        
        template <typename... TIntTypes>
        size_t IndexToOffset(TIntTypes... indexes) const
        {
            static_assert(sizeof...(TIntTypes) == uDimNum);
            size_t gap = 0;
            return NSShape::IndexToOffset(m_dims, gap, indexes...);
        }
        
        size_t operator[] (size_t idx) const
        {
            assert(idx < DimNum);
            return m_dims[idx];
        }
    private:
        std::array<size_t, uDimNum> m_dims{};
    };
    
    template <>
    class Shape<0>
    {
    public:
        constexpr static size_t DimNum = 0;

        explicit Shape() = default;
        
        bool operator == (const Shape& val) const
        {
            return true;
        }
        
        template <size_t vDimNum>
        bool operator == (const Shape<vDimNum>&) const
        {
            return false;
        }
        
        size_t Count() const
        {
            return 1;
        }
    };
    
    template <size_t v1, size_t v2>
    bool operator != (const Shape<v1> val1, const Shape<v2> val2)
    {
        return !(val1 == val2);
    }
}