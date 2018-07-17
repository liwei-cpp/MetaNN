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
namespace OperConv
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
        auto outSize = VarTypeDict<ConvParams::RowNum, ConvParams::ColNum>::Create()
                        .template Set<ConvParams::RowNum>(outRowNum)
                        .template Set<ConvParams::ColNum>(outColNum);

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
          std::enable_if_t<OperConv::valid<TInput, TKernel>>* = nullptr>
auto DefultConv(TInput&& p_input, TKernel&& p_kernel,
                TPadHeadValueCont&& p_padHead, TPadTailValueCont&& p_padTail,
                TStrideValueCont&& p_strides)
{
    return OperConv::DefaultEval(std::forward<TInput>(p_input), std::forward<TKernel>(p_kernel),
                                 std::forward<TPadHeadValueCont>(p_padHead),
                                 std::forward<TPadTailValueCont>(p_padTail),
                                 std::forward<TStrideValueCont>(p_strides));
}

template <typename TInput, typename TKernel,
          typename TStrideValueCont,
          std::enable_if_t<OperConv::valid<TInput, TKernel>>* = nullptr>
auto SameConv(TInput&& p_input, TKernel&& p_kernel,
              TStrideValueCont&& p_strides)
{
    return OperConv::SameEval(std::forward<TInput>(p_input), std::forward<TKernel>(p_kernel),
                              std::forward<TStrideValueCont>(p_strides));
}
}