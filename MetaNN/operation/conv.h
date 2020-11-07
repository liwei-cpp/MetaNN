#pragma once
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <cassert>
#include <type_traits>
#include <utility>
/*
namespace MetaNN::ConvParams
{
    struct RowNum;
    struct ColNum;
    struct PageNum;
}

namespace MetaNN
{
template <>
struct OperCategory_<ConvRelated::Conv2D,
                     CategoryTags::ThreeDArray,
                     CategoryTags::ThreeDArraySequence>
{
    using type = CategoryTags::ThreeDArray;
};


template <>
class OperAuxParams<ConvRelated::Conv2D, CategoryTags::ThreeDArray>
{
public:
    template <typename TPadHead, typename TPadTail, typename TStride>
    OperAuxParams(TPadHead&& head, TPadTail&& tail, TStride&& stride)
        : m_padHeadRow(head.template Get<ConvParams::RowNum>())
        , m_padHeadCol(head.template Get<ConvParams::ColNum>())
        , m_strideRow(stride.template Get<ConvParams::RowNum>())
        , m_strideCol(stride.template Get<ConvParams::ColNum>())
    {}
        
public:
    bool operator == (const OperAuxParams& val) const
    {
        return (m_padHeadRow == val.m_padHeadRow) &&
               (m_padHeadCol == val.m_padHeadCol) &&
               (m_strideRow == val.m_strideRow) &&
               (m_strideCol == val.m_strideCol);
    }
    
public:
    const size_t m_padHeadRow;
    const size_t m_padHeadCol;

    const size_t m_strideRow;
    const size_t m_strideCol;
};

template <>
class OperOrganizer<ConvRelated::Conv2D, CategoryTags::ThreeDArray>
{
    static size_t CalculateOutSize(size_t inSize,
                                   size_t padHead, size_t padTail,
                                   size_t stride, size_t kernelSize)
    {
        size_t tmp = inSize + padHead + padTail;
        if (tmp < kernelSize)
        {
            throw std::runtime_error("Input size is less than kernel size.");
        }
        
        auto res = (float)(tmp - kernelSize) / (float)(stride) + 1;
        return (size_t) res;
    }

public:
    template <typename TInput, typename TKernel,
              typename TPadHeadValueCont, typename TPadTailValueCont, 
              typename TStrideValueCont>
    OperOrganizer(TInput&& p_input, TKernel&& p_kernel,
                  TPadHeadValueCont&& p_padHead, TPadTailValueCont&& p_padTail,
                  TStrideValueCont&& p_strides)
        : m_rowNum(CalculateOutSize(p_input.RowNum(),
                                    p_padHead.template Get<ConvParams::RowNum>(),
                                    p_padTail.template Get<ConvParams::RowNum>(),
                                    p_strides.template Get<ConvParams::RowNum>(),
                                    p_kernel.RowNum()))
        , m_colNum(CalculateOutSize(p_input.ColNum(),
                                    p_padHead.template Get<ConvParams::ColNum>(),
                                    p_padTail.template Get<ConvParams::ColNum>(),
                                    p_strides.template Get<ConvParams::ColNum>(),
                                    p_kernel.ColNum()))
        , m_pageNum(p_kernel.Length())
    {}

    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }
    size_t PageNum() const { return m_pageNum; }

private:
    const size_t m_rowNum;
    const size_t m_colNum;
    const size_t m_pageNum;
};

namespace NSOperConv::NSCaseGen
{
template <typename TIn, typename TKernel, typename TElem, typename TDevice, typename TCategory>
class EvalUnit;

template <typename TIn, typename TKernel, typename TElem>
class EvalUnit<TIn, TKernel, TElem, DeviceTags::CPU, CategoryTags::ThreeDArray>
    : public BaseEvalUnit<DeviceTags::CPU>
{
    using CategoryType = CategoryTags::ThreeDArray;
public:
    EvalUnit(TIn input,
             TKernel kernel,
             OperAuxParams<ConvRelated::Conv2D, CategoryType> auxParams,
             OperOrganizer<ConvRelated::Conv2D, CategoryTags::ThreeDArray> org,
             EvalHandle<ThreeDArray<TElem, DeviceTags::CPU>> evalOutput)
        : m_input(std::move(input))
        , m_kernel(std::move(kernel))
        , m_auxParams(std::move(auxParams))
        , m_org(std::move(org))
        , m_evalOutput(std::move(evalOutput)) { }

    void Eval() override
    {
        const auto& input = m_input.Data();
        const auto& kernel = m_kernel.Data();
        
        assert(m_org.PageNum() == kernel.Length());

        m_evalOutput.Allocate(m_org.PageNum(), m_org.RowNum(), m_org.ColNum());
        auto& res = m_evalOutput.MutableData();
        
        for (size_t p = 0; p < m_org.PageNum(); ++p)
        {
            const auto& curKernel = kernel[p];
            
            for (size_t r = 0; r < m_org.RowNum(); ++r)
            {
                for (size_t c = 0; c< m_org.ColNum(); ++c)
                {
                    res.SetValue(p, r, c,
                                 CalculateItem(input, curKernel,
                                               r * m_auxParams.m_strideRow,
                                               c * m_auxParams.m_strideCol));
                }
            }
        }
        m_evalOutput.SetEval();
    }
    
private:
    auto CalculateItem(const ThreeDArray<TElem, DeviceTags::CPU>& input,
                       const ThreeDArray<TElem, DeviceTags::CPU>& curKernel,
                       size_t r, size_t c)
    {
        int xb = (int)r - (int)(m_auxParams.m_padHeadRow);
        int yb = (int)c - (int)(m_auxParams.m_padHeadCol);
        
        int xe = xb + (int)(curKernel.RowNum());
        int ye = yb + (int)(curKernel.ColNum());
                
        TElem res = TElem();
        for (size_t pc = 0; pc < input.PageNum(); ++pc)
        {
            for (int xi = xb; xi < xe; ++xi)
            {
                if ((xi < 0) || (xi >= (int)(input.RowNum()))) continue;
                for (int yi = yb; yi < ye; ++yi)
                {
                    if ((yi < 0) || (yi >= (int)(input.ColNum()))) continue;
                    res += input(pc, xi, yi) * curKernel(pc, xi - xb, yi - yb);
                }
            }
        }
        return res;
    }

private:
    TIn m_input;
    TKernel m_kernel;
    const OperAuxParams<ConvRelated::Conv2D, CategoryType> m_auxParams;
    const OperOrganizer<ConvRelated::Conv2D, CategoryTags::ThreeDArray> m_org;
    EvalHandle<ThreeDArray<TElem, DeviceTags::CPU>> m_evalOutput;
};

struct Calculator
{
    template <typename TCaseTail, typename TEvalRes, typename TOper>
    static void EvalRegister(TEvalRes& evalRes, const TOper& oper)
    {
        
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;
        using CategoryType = DataCategory<typename TEvalRes::DataType>;

        auto inputHandle = oper.Operand1().EvalRegister();
        auto kernelHandle = oper.Operand2().EvalRegister();
        
        using UnitType = EvalUnit<decltype(inputHandle), decltype(kernelHandle),
                                  ElementType, DeviceType, CategoryType>;
        using GroupType = TrivialEvalGroup<UnitType>;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        auto depVec = {inputHandle.DataPtr(), kernelHandle.DataPtr()};
        
        UnitType unit(std::move(inputHandle), std::move(kernelHandle),
                      oper.AuxParams(),
                      oper.Ogranizer(),
                      std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, std::move(depVec));

    }
};
}

namespace NSOperConv
{    
    inline size_t CalculatePadSize(size_t inSize,
                                   size_t stride, size_t kernelSize)
    {
        size_t tmp = (inSize + stride - 1) / stride;
        return (tmp - 1) * stride + kernelSize - inSize;
    }
    
    
    template <typename TInput, typename TKernel>
    constexpr bool valid = (IsThreeDArray<TInput> && IsThreeDArraySequence<TKernel>);
    
/// Convolution with "Default" padding mode
    // 3D-Array conv, commonly used for image convolution
    template<typename TInput, typename TKernel,
             typename TPadHeadValueCont, typename TPadTailValueCont, 
             typename TStrideValueCont,
             std::enable_if_t<IsThreeDArray<TInput>>* = nullptr,
             std::enable_if_t<IsThreeDArraySequence<TKernel>>* = nullptr>
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

        using ResType = BinaryOp<ConvRelated::Conv2D,
                                 RemConstRef<TInput>,
                                 RemConstRef<TKernel>>;
        return ResType(std::forward<TInput>(p_input), std::forward<TKernel>(p_kernel),
                       std::forward<TPadHeadValueCont>(p_padHead),
                       std::forward<TPadTailValueCont>(p_padTail),
                       std::forward<TStrideValueCont>(p_strides));
    }
    
/// Convolution with "Same" padding mode
    template<typename TInput, typename TKernel,
             typename TStrideValueCont,
             std::enable_if_t<IsThreeDArray<TInput>>* = nullptr,
             std::enable_if_t<IsThreeDArraySequence<TKernel>>* = nullptr>
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
auto DefaultConv(TInput&& p_input, TKernel&& p_kernel,
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
}*/