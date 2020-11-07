#pragma once
#include <MetaNN/facilities/_.h>
#include <MetaNN/operation/facilities/_.h>
#include <MetaNN/policies/_.h>
#include <MetaNN/facilities/cont_metafuns/value_sequential.h>
#include <numeric>

namespace MetaNN::OpTags
{
    struct ReduceSum;
}

namespace MetaNN
{
    namespace OperReduceSum
    {
        template <typename TPDim, typename TIndexes>
        struct DimArrToBitHelper_;
        
        template <typename TPDim, size_t... I>
        struct DimArrToBitHelper_<TPDim, std::index_sequence<I...>>
        {
            using type = PDimBitArrayIs<ValueSequential::Contains<TPDim, I>...>;
        };

        template <typename TPolicy, size_t uDimNum>
        struct DimArrToBit_
        {
            using PDim = PickPolicyOjbect<TPolicy, DimPolicy, DimPolicy::DimArrayValueCate>;
            using type = typename DimArrToBitHelper_<PDim, std::make_index_sequence<uDimNum>>::type;
        };

        template <size_t uTrueBound, typename TIndexes>
        struct ModDimToBitHelper_;
        
        template <size_t uTrueBound, size_t... I>
        struct ModDimToBitHelper_<uTrueBound, std::index_sequence<I...>>
        {
            using type = PDimBitArrayIs<(I < uTrueBound)...>;
        };

        template <typename TPolicy, size_t uDimNum>
        struct ModDimToBit_
        {
            constexpr static size_t ModDimNum = PolicySelect<DimPolicy, TPolicy>::ModifyDimNum;
            static_assert(ModDimNum <= uDimNum);
            using type = typename ModDimToBitHelper_<ModDimNum, std::make_index_sequence<uDimNum>>::type;
        };

        template <typename TIndexes>
        struct DefaultToBitHelper_;
        
        template <size_t... I>
        struct DefaultToBitHelper_<std::index_sequence<I...>>
        {
            using type = PDimBitArrayIs<(I >= 0)...>;
        };

        template <size_t uDimNum>
        struct DefaultToBit_
        {
            using type = typename DefaultToBitHelper_<std::make_index_sequence<uDimNum>>::type;
        };

        template <size_t uDimNum>
        constexpr bool IsTrivialDimBits(std::array<bool, uDimNum> dimBits)
        {
            for (size_t i = 0; i < uDimNum; ++i)
            {
                if (dimBits[i]) return false;
            }
            return true;
        }

        template <size_t uDimNum>
        constexpr size_t AccuDimBits(std::array<bool, uDimNum> dimBits)
        {
            size_t res = 0;
            for (size_t i = 0; i < uDimNum; ++i)
            {
                if (dimBits[i]) ++res;
            }
            return res;
        }
        
        template <template<auto...> class Cont, auto DimBits, typename>
        struct DimBits2VaridicTemp_;
        
        template <template<auto...> class Cont, auto DimBits, int... Is>
        struct DimBits2VaridicTemp_<Cont, DimBits, Helper::IndexSequence<Is...>>
        {
            using type = Cont<DimBits[Is]...>;
        };
    }
    
    namespace OperReduceSum::NSCaseGen
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
            constexpr static size_t OriDim = CategoryTagFromHandle<TInputHandle>::DimNum;

        protected:
            virtual void EvalInternalLogic(EvalItemType& evalItem) final override
            {
                const auto& in = evalItem.m_inputHandle.Data();
                const auto& oriShape = in.Shape();
                
                constexpr auto dimBits = PolicySelect<DimPolicy, TPolicies>::DimBitArray;
                static_assert(OriDim == dimBits.size());
                constexpr size_t AimDim = CategoryTagFromHandle<TOutputHandle>::DimNum;
                
                using ResType = typename TOutputHandle::DataType;
                using ElementType = typename ResType::ElementType;
                ResType out(evalItem.m_outShape);

                const size_t count = oriShape.Count();
                if (count == 0)
                {
                    evalItem.m_outputHandle.SetData(std::move(out));
                    return;
                }
                
                auto low_in = LowerAccess(in);
                const ElementType* mem_in = low_in.RawMemory();
                auto low_out = LowerAccess(out);
                ElementType* mem_out = low_out.MutableRawMemory();
                
                static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
                if constexpr (AimDim == 0)
                {
                    mem_out[0] = std::accumulate(mem_in, mem_in + count, ElementType{});
                }
                else if constexpr (IsRegular(dimBits))
                {
                    const size_t new_count = evalItem.m_outShape.Count();
                    assert(count % new_count == 0);
                    std::copy(mem_in, mem_in + new_count, mem_out);
                    for (size_t i = new_count; i < count; i += new_count)
                    {
                        std::transform(mem_in + i, mem_in + i + new_count, mem_out, mem_out, std::plus<ElementType>());
                    }
                }
                else
                {
                    std::fill(mem_out, mem_out + evalItem.m_outShape.Count(), ElementType{});
                    mem_out[0] = mem_in[0];

                    std::array<size_t, OriDim> oriPos{};
                    std::array<size_t, AimDim> aimPos{};
                    for (size_t i = 1; i < count; ++i)
                    {
                        oriShape.ShiftIndex(oriPos);
                        if constexpr (PolicySelect<DimPolicy, TPolicies>::IsKeepDim)
                        {
                            for (size_t j = 0; j < OriDim; ++j)
                            {
                                aimPos[j] = dimBits[j] ? 0 : oriPos[j];
                            }
                        }
                        else
                        {
                            size_t p = 0;
                            for (size_t j = 0; j < OriDim; ++j)
                            {
                                if (!dimBits[j])
                                {
                                    aimPos[p++] = oriPos[j];
                                }
                            }
                        }
                        mem_out[evalItem.m_outShape.IndexToOffset(aimPos)] += mem_in[i];
                    }
                }
                evalItem.m_outputHandle.SetData(std::move(out));
            }
        private:
            constexpr static bool IsRegular(std::array<bool, OriDim> dimBits)
            {
                size_t i = 0;
                for (; i < OriDim; ++i)
                {
                    if (dimBits[i] == 0) break;
                }

                for (; i < OriDim; ++i)
                {
                    if (dimBits[i] == 1) return false;
                }
                return true;
            }
        };
    }

    template <>
    struct OperSeq_<OpTags::ReduceSum>
    {
        using type = OperCalAlgoChain<TailCalculator<OperReduceSum::NSCaseGen::EvalItem,
                                                     OperReduceSum::NSCaseGen::EvalGroup,
                                                     PolicyContainer<PPassPolicy, PPassShape>>>;
    };

    template <typename TCate, typename TPolicies>
    class OperShapeInfo<OpTags::ReduceSum, TCate, TPolicies>
    {
        constexpr static size_t uDim = TCate::DimNum;
    public:
        template <typename TOperAuxParams, typename TOperand>
        OperShapeInfo(const TOperAuxParams&, const TOperand& operand)
        {
            if constexpr(uDim != 0)
            {
                constexpr auto dimBits = PolicySelect<DimPolicy, TPolicies>::DimBitArray;
                auto prevShape = operand.Shape();
                if constexpr (PolicySelect<DimPolicy, TPolicies>::IsKeepDim)
                {
                    for (size_t i = 0; i < dimBits.size(); ++i)
                    {
                        m_shape[i] = dimBits[i] ? 1 : prevShape[i];
                    }
                }
                else
                {
                    size_t p = 0;
                    for (size_t i = 0; i < dimBits.size(); ++i)
                    {
                        if (!dimBits[i])
                        {
                            m_shape[p++] = prevShape[i];
                        }
                    }
                    assert(p == uDim);
                }
            }
        }
    
        const auto& Shape() const
        {
            return m_shape;
        }
    
    private:
        MetaNN::Shape<uDim> m_shape;
    };
    
    template <typename TPolicy, typename TOperand>
    struct OperCategory_<OpTags::ReduceSum, TPolicy, TOperand>
    {
        constexpr static size_t OriDim = TOperand::DimNum;
        constexpr static auto DimBits = PolicySelect<DimPolicy, TPolicy>::DimBitArray;
        constexpr static size_t AccuBits = 
            PolicySelect<DimPolicy, TPolicy>::IsKeepDim ?
            0 : OperReduceSum::AccuDimBits(DimBits);
        static_assert(OriDim >= AccuBits);
        using type = CategoryTags::Tensor<OriDim - AccuBits>;
    };
    
    template <typename TP>
    constexpr bool IsValidOper<OpTags::ReduceSum, TP> = (IsValidCategoryTag<DataCategory<TP>>) &&
                                                        (DataCategory<TP>::DimNum > 0);
    
    template <typename TPolicy = PolicyContainer<>, typename TP,
              std::enable_if_t<IsValidOper<OpTags::ReduceSum, TP>>* = nullptr>
    auto ReduceSum(TP&& oper)
    {
        constexpr bool HasDimArray = HasNonTrivialPolicy<TPolicy, DimPolicy, DimPolicy::DimArrayValueCate>;
        constexpr bool HasModDimNum = HasNonTrivialPolicy<TPolicy, DimPolicy, DimPolicy::ModifyDimNumValueCate>;

        static_assert((int)HasDimArray + (int)HasModDimNum <= 1, "only one of DimArray or ResDimValue could be set");
        
        using TDimBits = typename CompileTimeSwitch<std::integer_sequence<bool, HasDimArray, HasModDimNum>,
                                                    std::tuple<OperReduceSum::DimArrToBit_<TPolicy, DataCategory<TP>::DimNum>,
                                                               OperReduceSum::ModDimToBit_<TPolicy, DataCategory<TP>::DimNum>,
                                                               OperReduceSum::DefaultToBit_<DataCategory<TP>::DimNum>>>::type;

        static constexpr bool KeepDim = PolicySelect<DimPolicy, TPolicy>::IsKeepDim;
        constexpr auto DimBits = TDimBits::DimBitArray;
        if constexpr (OperReduceSum::IsTrivialDimBits(DimBits))
        {
            return std::forward<TP>(oper);
        }
        else
        {
            using rawOp = RemConstRef<TP>;
            using ResType = Operation<OpTags::ReduceSum, OperandContainer<rawOp>,
                                      PolicyContainer<TDimBits, PKeepDimValueIs<KeepDim>>>;
            return ResType(std::forward<TP>(oper));
        }
    }
}