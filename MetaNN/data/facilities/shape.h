#pragma once
#include <MetaNN/data/facilities/category_tags.h>
#include <MetaNN/facilities/cont_metafuns/helpers.h>
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
        
        template <size_t uDimNum, int...N>
        size_t IndexToOffset(const std::array<size_t, uDimNum>& dims,
                             const std::array<size_t, uDimNum>& indexes,
                             Helper::IndexSequence<N...>*)
        {
            size_t gap = 0;
            return IndexToOffset(dims, gap, std::get<N>(indexes)...);
        }

        template <size_t ID, typename TShape>
        void FillShape(TShape& pShape)
        {
            return;
        }

        template <size_t ID, typename TShape, typename TCurParam, typename... TShapeParameter>
        void FillShape(TShape& pShape, TCurParam curParam, TShapeParameter... shapes)
        {
            pShape[ID] = static_cast<size_t>(curParam);
            FillShape<ID + 1>(pShape, shapes...);
        }
        
    }

    template <size_t uDimNum>
    class Shape
    {
        static_assert(uDimNum > 0);
    public:
        constexpr static size_t DimNum = uDimNum;

        explicit Shape() = default;
        
        template <typename... TIntTypes,
                  typename = std::enable_if_t<(std::is_convertible_v<TIntTypes, size_t> && ...)>>
        explicit Shape(TIntTypes... shapes)
        {
            static_assert(sizeof...(TIntTypes) == uDimNum);
            NSShape::FillShape<0>(m_dims, shapes...);
        }
        
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
        
        template <typename... TIntTypes,
                  typename = std::enable_if_t<(std::is_convertible_v<TIntTypes, size_t> && ...)>>
        size_t IndexToOffset(TIntTypes... indexes) const
        {
            static_assert(sizeof...(TIntTypes) == uDimNum);
            size_t gap = 0;
            return NSShape::IndexToOffset(m_dims, gap, indexes...);
        }

        size_t IndexToOffset(const std::array<size_t, DimNum>& indexes) const
        {
            using TSeq = Helper::MakeIndexSequence<DimNum>;
            return NSShape::IndexToOffset(m_dims, indexes, (TSeq*)nullptr);
        }
        
        std::array<size_t, DimNum> OffsetToIndex(size_t offset) const
        {
            std::array<size_t, DimNum> res;
            for (int i = (int)DimNum - 1; i >= 0; --i)
            {
                res[i] = offset % m_dims[i];
                offset = (offset - res[i]) / m_dims[i];
            }
            if (offset != 0)
            {
                throw std::runtime_error("Offset out of bound!");
            }
            return res;
        }
        
        void ShiftIndex(std::array<size_t, DimNum>& indexes, int carry = 1) const
        {
            if (carry == 0) return;
            if (carry > 0)
            {
                size_t uCarry = (size_t)carry;
                for (int i = (int)DimNum - 1; i >= 0; --i)
                {
                    indexes[i] += uCarry;
                    uCarry = indexes[i] / m_dims[i];
                    indexes[i] %= m_dims[i];
                    if (uCarry == 0) return;
                }
                if (uCarry)
                {
                    throw std::runtime_error("Overflow");
                }
            }
            else
            {
                throw std::runtime_error("Not implemented yet.");
            }
        }

        size_t operator[] (size_t idx) const
        {
            assert(idx < DimNum);
            return m_dims[idx];
        }
        
        size_t& operator[] (size_t idx)
        {
            assert(idx < DimNum);
            return m_dims[idx];
        }
        
        decltype(auto) begin()
        {
            return m_dims.begin();
        }
        
        decltype(auto) begin() const
        {
            return m_dims.begin();
        }
        
        decltype(auto) end()
        {
            return m_dims.end();
        }
        
        decltype(auto) end() const
        {
            return m_dims.end();
        }
        
        decltype(auto) rbegin()
        {
            return m_dims.rbegin();
        }
        
        decltype(auto) rbegin() const
        {
            return m_dims.rbegin();
        }
        
        decltype(auto) rend()
        {
            return m_dims.rend();
        }
        
        decltype(auto) rend() const
        {
            return m_dims.rend();
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

    template <typename... TShapeParameter>
    explicit Shape(TShapeParameter... shapes) -> Shape<sizeof...(TShapeParameter)>;
}