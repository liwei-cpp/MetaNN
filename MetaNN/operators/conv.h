#pragma once
#include <MetaNN/data/scalar.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/matrices/trival_matrix.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/operators/parameters/conv_params.h>
#include <cassert>
#include <type_traits>
#include <utility>

namespace MetaNN
{
template <>
class OperAuxParams<ConvRelated::Conv2D, CategoryTags::ThreeDArray>
{
public:
    template <typename TPadHead, typename TPadTail, typename TStride>
    OperAuxParams(TPadHead&& head, TPadTail&& tail, TStride&& stride)
        : m_padHeadRow(head.template Get<ConvParams::RowNum>())
        , m_padHeadCol(head.template Get<ConvParams::ColNum>())
        , m_padHeadPage(head.template Get<ConvParams::PageNum>())
        , m_padTailRow(tail.template Get<ConvParams::RowNum>())
        , m_padTailCol(tail.template Get<ConvParams::ColNum>())
        , m_padTailPage(tail.template Get<ConvParams::PageNum>())
        , m_strideRow(stride.template Get<ConvParams::RowNum>())
        , m_strideCol(stride.template Get<ConvParams::ColNum>())
    {}
        
public:
    bool operator == (const OperAuxParams& val) const
    {
        return (m_padHeadRow == val.m_padHeadRow) &&
               (m_padHeadCol == val.m_padHeadCol) &&
               (m_padHeadPage == val.m_padHeadPage) &&
               (m_padTailRow == val.m_padTailRow) &&
               (m_padTailCol == val.m_padTailCol) &&
               (m_padTailPage == val.m_padTailPage) &&
               (m_strideRow == val.m_strideRow) &&
               (m_strideCol == val.m_strideCol);
    }
    
public:
    const size_t m_padHeadRow;
    const size_t m_padHeadCol;
    const size_t m_padHeadPage;
    
    const size_t m_padTailRow;
    const size_t m_padTailCol;
    const size_t m_padTailPage;
    
    const size_t m_strideRow;
    const size_t m_strideCol;
};

namespace NSOperConv::NSCaseGen
{
struct Calculator
{
    template <typename TCaseTail, typename TEvalRes, typename TOperator1, typename TOperator2>
    static void EvalRegister(TEvalRes& evalRes, const TOperator1& oper1, const TOperator2& oper2)
    {
        /*
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;
        using CategoryType = DataCategory<typename TEvalRes::DataType>;

        auto handle1 = oper1.EvalRegister();
        auto handle2 = oper2.EvalRegister();
        using UnitType = EvalUnit<decltype(handle1), decltype(handle2),
                                  ElementType, DeviceType, CategoryType>;
        using GroupType = TrivalEvalGroup<UnitType>;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        auto depVec = {handle1.DataPtr(), handle2.DataPtr()};
        
        UnitType unit(std::move(handle1), std::move(handle2), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, std::move(depVec));
        */
    }
};
}

template <>
struct OperSeq_<ConvRelated::Conv2D>
{
    using type = OperSeqContainer<NSOperConv::NSCaseGen::Calculator>;
};

namespace NSOperConv
{
    inline size_t CalculateOutSize(size_t inSize,
                                   size_t padHead, size_t padTail,
                                   size_t stride, size_t kernelSize)
    {
        size_t tmp = inSize + padHead + padTail;
        if (tmp < kernelSize)
        {
            throw std::runtime_error("Input size is less than kernel size.");
        }
        
        auto res = (float)(inSize - kernelSize) / (float)(stride) + 1;
        return (size_t) res;
    }
    
    inline size_t CalculatePadSize(size_t inSize,
                                   size_t stride, size_t kernelSize)
    {
        size_t tmp = (inSize + stride - 1) / stride;
        return (tmp - 1) * stride + kernelSize - inSize;
    }
    
    
    template <typename TInput, typename TKernel>
    constexpr bool valid = (IsThreeDArray<TInput> && IsThreeDArray<TKernel>);
    
/// Convolution with "Default" padding mode
    // 3D-Array conv, commonly used for image convolution
    template<typename TInput, typename TKernel,
             typename TPadHeadValueCont, typename TPadTailValueCont, 
             typename TStrideValueCont,
             std::enable_if_t<IsThreeDArray<TInput>>* = nullptr,
             std::enable_if_t<IsThreeDArray<TKernel>>* = nullptr>
    static auto DefaultEval(TInput&& p_input, TKernel&& p_kernel,
                            TPadHeadValueCont&& p_padHead, TPadTailValueCont&& p_padTail,
                            TStrideValueCont&& p_strides)
    {
        using rawInput = RemConstRef<TInput>;
        using rawKernel = RemConstRef<TKernel>;
        
        using ElementType = typename rawInput::ElementType;
        using DeviceType = typename rawInput::DeviceType;
        static_assert(std::is_same_v<ElementType, typename rawKernel::ElementType>,
                      "Different element types cannot conv directly");
        static_assert(std::is_same_v<DeviceType, typename rawKernel::DeviceType>,
                      "Different device types cannot conv directly");

        if (p_input.PageNum() != p_kernel.PageNum())
        {
            throw std::runtime_error("The input and kernel should have same depth!");
        }


        size_t outRowNum = CalculateOutSize(p_input.RowNum(),
                                            p_padHead.template Get<ConvParams::RowNum>(),
                                            p_padTail.template Get<ConvParams::RowNum>(),
                                            p_strides.template Get<ConvParams::RowNum>(),
                                            p_kernel.RowNum());
        size_t outColNum = CalculateOutSize(p_input.ColNum(),
                                            p_padHead.template Get<ConvParams::ColNum>(),
                                            p_padTail.template Get<ConvParams::ColNum>(),
                                            p_strides.template Get<ConvParams::ColNum>(),
                                            p_kernel.ColNum());
        auto outSize = VarTypeDict<ConvParams::RowNum,
                                   ConvParams::ColNum,
                                   ConvParams::PageNum>::Create()
                        .template Set<ConvParams::RowNum>(outRowNum)
                        .template Set<ConvParams::ColNum>(outColNum)
                        .template Set<ConvParams::PageNum>(p_input.PageNum());

        return MakeConvTemplate(std::forward<TInput>(p_input), std::forward<TKernel>(p_kernel),
                                std::forward<TPadHeadValueCont>(p_padHead),
                                std::forward<TPadTailValueCont>(p_padTail),
                                std::forward<TStrideValueCont>(p_strides),
                                std::move(outSize));
    }
    
/// Convolution with "Same" padding mode
    // 3D-Array conv, commonly used for image convolution
    template<typename TInput, typename TKernel,
             typename TStrideValueCont,
             std::enable_if_t<IsThreeDArray<TInput>>* = nullptr,
             std::enable_if_t<IsThreeDArray<TKernel>>* = nullptr>
    static auto SameEval(TInput&& p_input, TKernel&& p_kernel,
                         TStrideValueCont&& p_strides)
    {
        const size_t rowPad = CalculatePadSize(p_input.RowNum(), p_strides.template Get<ConvParams::RowNum>(), p_kernel.RowNum());
        const size_t colPad = CalculatePadSize(p_input.ColNum(), p_strides.template Get<ConvParams::ColNum>(), p_kernel.ColNum());
        
        const size_t rowPadHead = rowPad / 2;
        const size_t colPadHead = colPad / 2;
        
        const size_t rowPadTail = rowPad - rowPadHead;
        const size_t colPadTail = colPad - colPadHead;
        
        auto padHead = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                        .template Set<ConvParams::RowNum>(rowPadHead)
                        .template Set<ConvParams::ColNum>(colPadHead);
        auto padTail = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                        .template Set<ConvParams::RowNum>(rowPadTail)
                        .template Set<ConvParams::ColNum>(colPadTail);
        return DefaultEval(std::forward<TInput>(p_input), std::forward<TKernel>(p_kernel),
                           std::move(padHead), std::move(padTail),
                           std::forward<TStrideValueCont>(p_strides));
    }
};

template <typename TInput, typename TKernel,
          typename TPadHeadValueCont, typename TPadTailValueCont, 
          typename TStrideValueCont,
          std::enable_if_t<NSOperConv::valid<TInput, TKernel>>* = nullptr>
auto DefultConv(TInput&& p_input, TKernel&& p_kernel,
                TPadHeadValueCont&& p_padHead, TPadTailValueCont&& p_padTail,
                TStrideValueCont&& p_strides)
{
    return NSOperConv::DefaultEval(std::forward<TInput>(p_input), std::forward<TKernel>(p_kernel),
                                 std::forward<TPadHeadValueCont>(p_padHead),
                                 std::forward<TPadTailValueCont>(p_padTail),
                                 std::forward<TStrideValueCont>(p_strides));
}

template <typename TInput, typename TKernel,
          typename TStrideValueCont,
          std::enable_if_t<NSOperConv::valid<TInput, TKernel>>* = nullptr>
auto SameConv(TInput&& p_input, TKernel&& p_kernel,
              TStrideValueCont&& p_strides)
{
    return NSOperConv::SameEval(std::forward<TInput>(p_input), std::forward<TKernel>(p_kernel),
                                std::forward<TStrideValueCont>(p_strides));
}
}