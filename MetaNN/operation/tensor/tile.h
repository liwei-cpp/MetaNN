#pragma once
#include <MetaNN/operation/facilities/_.h>
#include <MetaNN/policies/_.h>
#include <MetaNN/facilities/cont_metafuns/value_sequential.h>
#include <MetaNN/facilities/_.h>
#include <cassert>

namespace MetaNN::OpTags
{
    struct Tile;
}

namespace MetaNN
{
    namespace OperTile
    {
        template <typename TIndexes, typename TDimArray>
        struct DimArrToBitHelper_;
        
        template <typename TDimArray, size_t... I>
        struct DimArrToBitHelper_<std::index_sequence<I...>, TDimArray>
        {
            using type = PDimBitArrayIs<ValueSequential::Contains<TDimArray, I>...>;
        };
        
        template <size_t AimDim, typename TPolicy>
        struct DimArrToBit_
        {
            using DimArray = PickPolicyOjbect<TPolicy, DimPolicy, DimPolicy::DimArrayValueCate>;
            using type = typename DimArrToBitHelper_<std::make_index_sequence<AimDim>, DimArray>::type;
        };
        
        template <typename TIndexes, size_t TrueDimBound>
        struct DefaultToBitHelper_;
        
        template <size_t TrueDimBound, size_t... I>
        struct DefaultToBitHelper_<std::index_sequence<I...>, TrueDimBound>
        {
            using type = PDimBitArrayIs<(I < TrueDimBound)...>;
        };
        
        template <size_t AimDim, size_t OriDim>
        struct DefaultToBit_
        {
            using type = typename DefaultToBitHelper_<std::make_index_sequence<AimDim>, AimDim - OriDim>::type;
        };

        template <size_t AimDim, size_t uDim>
        constexpr bool CheckBound(std::array<size_t, uDim> dimArray)
        {
            for (size_t i = 0; i < uDim; ++i)
            {
                if (dimArray[i] >= AimDim)
                {
                    return false;
                }
            }
            return true;
        }

        template <size_t uAimDim, size_t uOriDim>
        auto GetMuteShape(const std::array<bool, uAimDim>& dimBitArray, const Shape<uOriDim>& shape)
        {
            Shape<uAimDim> resShape;
            
            size_t oriPos = 0;
            for (size_t aimPos = 0; aimPos < uAimDim; ++aimPos)
            {
                if (dimBitArray[aimPos])
                {
                    resShape[aimPos] = 1;
                }
                else
                {
                    resShape[aimPos] = shape[oriPos++];
                }
            }
            
            if (oriPos != uOriDim)
            {
                throw std::runtime_error("GetMuteShape error, there might be duplicate array num in the input policy.");;
            }
            return resShape;
        }

        template <size_t uDim>
        bool IsShapeCompatible(const Shape<uDim>& outShape, const Shape<uDim>& inShape)
        {
            for (size_t i = 0; i < uDim; ++i)
            {
                if (outShape[i] % inShape[i] != 0)
                    return false;
            }
            return true;
        }
    }
    
    namespace OperTile::NSCaseGen
    {
        template <typename TInputHandle, typename TOutputHandle, typename TPolicies>
        class EvalItem : public BaseEvalItem
        {
        public:
            using CategoryTag = CategoryTagFromHandle<TOutputHandle>;

            EvalItem(TInputHandle oriHandle, TOutputHandle outputHandle, Shape<CategoryTag::DimNum> shape)
                : BaseEvalItem(TypeID<EvalItem>(),
                               {oriHandle.DataPtr()}, outputHandle.DataPtr())
                , m_inputHandle(std::move(oriHandle))
                , m_outputHandle(std::move(outputHandle))
                , m_outShape(std::move(shape))
            {}
        
            const TInputHandle m_inputHandle;
            TOutputHandle m_outputHandle;
            Shape<CategoryTag::DimNum> m_outShape;
        };
        
        template <typename TInputHandle, typename TOutputHandle, typename TPolicies>
        class EvalGroup : public TrivialEvalGroup<EvalItem<TInputHandle, TOutputHandle, TPolicies>>
        {
            using EvalItemType = EvalItem<TInputHandle, TOutputHandle, TPolicies>;
            constexpr static size_t AimDim = CategoryTagFromHandle<TOutputHandle>::DimNum;
            constexpr static auto dimBits = PolicySelect<DimPolicy, TPolicies>::DimBitArray;

        protected:
            virtual void EvalInternalLogic(EvalItemType& evalItem) final override
            {
                if (evalItem.m_outShape.Count() == 0)
                {
                    using ResType = typename TOutputHandle::DataType;
                    ResType out(evalItem.m_outShape);
                    evalItem.m_outputHandle.SetData(std::move(out));
                    return;
                }
                if constexpr (IsRegular(dimBits))
                {
                    const auto& in = evalItem.m_inputHandle.Data();
                    if (IsBroadcastMatch(in.Shape(), evalItem.m_outShape))
                    {
                        EvalRegular(evalItem);
                    }
                    else
                    {
                        EvalGen(evalItem);
                    }
                }
                else
                {
                    EvalGen(evalItem);
                }
            }
        private:
            constexpr static bool IsRegular(std::array<bool, AimDim> dimBits)
            {
               size_t i = 0;
                for (; i < AimDim; ++i)
                {
                    if (dimBits[i] == 0) break;
                }

                for (; i < AimDim; ++i)
                {
                    if (dimBits[i] == 1) return false;
                }
                return true;
            }
            
            void EvalRegular(EvalItemType& evalItem)
            {
                const auto& in = evalItem.m_inputHandle.Data();

                using ResType = typename TOutputHandle::DataType;
                using ElementType = typename ResType::ElementType;
                ResType out(evalItem.m_outShape);
                
                static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");

                auto low_in = LowerAccess(in);
                const ElementType* mem_in = low_in.RawMemory();
                auto low_out = LowerAccess(out);
                ElementType* mem_out = low_out.MutableRawMemory();
                
                const size_t aimCount = evalItem.m_outShape.Count();
                const size_t oriCount = in.Shape().Count();
                assert(aimCount % oriCount == 0);

                for (size_t i = 0; i < aimCount / oriCount; ++i)
                {
                    std::copy(mem_in, mem_in + oriCount, mem_out);
                    mem_out += oriCount;
                }
                evalItem.m_outputHandle.SetData(std::move(out));
            }
            
            void EvalGen(EvalItemType& evalItem)
            {
                const auto& in = evalItem.m_inputHandle.Data();
                const auto& muteShape = OperTile::GetMuteShape(dimBits, in.Shape());
                static_assert(RemConstRef<decltype(muteShape)>::DimNum == AimDim);

                using ResType = typename TOutputHandle::DataType;
                using ElementType = typename ResType::ElementType;
                ResType out(evalItem.m_outShape);
                
                static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");

                auto low_in = LowerAccess(in);
                const ElementType* mem_in = low_in.RawMemory();
                auto low_out = LowerAccess(out);
                ElementType* mem_out = low_out.MutableRawMemory();
                mem_out[0] = mem_in[0];

                const size_t count = evalItem.m_outShape.Count();
                std::array<size_t, AimDim> oriPos{};
                std::array<size_t, AimDim> projPos{};

                for (size_t i = 1; i < count; ++i)
                {
                    evalItem.m_outShape.ShiftIndex(oriPos);
                    for (size_t j = 0; j < AimDim; ++j)
                        projPos[j] = oriPos[j] % muteShape[j];
                    size_t index = muteShape.IndexToOffset(projPos);
                    mem_out[i] = mem_in[index];
                }
                evalItem.m_outputHandle.SetData(std::move(out));
            }
        };
    }
    
    template <>
    struct OperSeq_<OpTags::Tile>
    {
        using type = OperCalAlgoChain<TailCalculator<OperTile::NSCaseGen::EvalItem,
                                                     OperTile::NSCaseGen::EvalGroup,
                                                     PolicyContainer<PPassPolicy, PPassShape>>>;
    };

    template <typename TPolicy, typename TOp>
    struct OperCategory_<OpTags::Tile, TPolicy, TOp>
    {
        constexpr static auto dimBitArray = PolicySelect<DimPolicy, TPolicy>::DimBitArray;
        using type = CategoryTags::Tensor<dimBitArray.size()>;
    };

    template <typename TElem, typename TCate>
    class OperAuxParams<OpTags::Tile, TElem, TCate>
    {
    public:
        OperAuxParams(Shape<TCate::DimNum> aimShape)
            : m_shape(std::move(aimShape))
        {}

        const auto& GetShape() const { return m_shape; }
        bool operator == (const OperAuxParams& val) const
        {
            return m_shape == val.m_shape;
        }
    private:
        Shape<TCate::DimNum> m_shape;
    };
    
    template <typename TCate, typename TPolicy>
    class OperShapeInfo<OpTags::Tile, TCate, TPolicy>
    {
    public:
        template <typename TOperAuxParams, typename TOp>
        OperShapeInfo(const TOperAuxParams& param, const TOp& op)
            : m_shape(param.GetShape())
        { }

        const auto& Shape() const
        {
            return m_shape;
        }

    private:
        MetaNN::Shape<TCate::DimNum> m_shape;
    };

    template <typename TPolicy = PolicyContainer<>, typename TP, size_t AimDim,
              std::enable_if_t<IsValidOper<OpTags::Tile, TP>>* = nullptr>
    auto Tile(TP&& oper, Shape<AimDim> aimShape)
    {
        static_assert(DataCategory<TP>::DimNum <= AimDim);

        constexpr bool HasDimArray = HasNonTrivialPolicy<TPolicy, DimPolicy, DimPolicy::DimArrayValueCate>;
        if constexpr (HasNonTrivialPolicy<TPolicy, DimPolicy, DimPolicy::DimArrayValueCate>)
        {
            constexpr auto dimArray = PolicySelect<DimPolicy, TPolicy>::DimArray;
            static_assert(DataCategory<TP>::DimNum + dimArray.size() == AimDim,
                          "Shape size mismatch");
        }

        using TDimBits = typename std::conditional_t<HasDimArray,
                                                     OperTile::DimArrToBit_<AimDim, TPolicy>,
                                                     OperTile::DefaultToBit_<AimDim, DataCategory<TP>::DimNum>>::type;
        assert(OperTile::IsShapeCompatible(aimShape,
                                           OperTile::GetMuteShape(TDimBits::DimBitArray, oper.Shape())));

        using rawOp = RemConstRef<TP>;
        using ResType = Operation<OpTags::Tile, OperandContainer<rawOp>,
                                  PolicyContainer<TDimBits>>;
        return ResType(OperAuxParams<OpTags::Tile,
                                     typename rawOp::ElementType,
                                     CategoryTags::Tensor<AimDim>>(std::move(aimShape)),
                       std::forward<TP>(oper));
    }
}