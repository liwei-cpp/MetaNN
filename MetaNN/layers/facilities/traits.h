#pragma once

#include <stack>
#include <type_traits>
#include <MetaNN/facilities/var_type_dict.h>
#include <MetaNN/layers/facilities/layer_io_map.h>

namespace MetaNN::LayerTraits
{
/*
template <typename TWeight, typename TGrad, typename TGradCollector>
void MatrixGradCollect(const TWeight& weight,
                       TGrad& grad,
                       TGradCollector& col)
{
    while (!grad.empty())
    {
        auto g = grad.top();
        grad.pop();
        col.Collect(weight, g);
    }
}
*/
namespace NSLayerIOMapTrasfer
{
template <typename TVarTypeDict, typename... TKVs>
struct CreateVarTypeDict_
{
    using type = TVarTypeDict;
};

template <template<typename...> class TVarTypeDictCont, typename... TProcessed, typename TCur, typename... TKVs>
struct CreateVarTypeDict_<TVarTypeDictCont<TProcessed...>, TCur, TKVs...>
{
    using type = typename CreateVarTypeDict_<TVarTypeDictCont<TProcessed...,
                                                              typename TCur::KeyType>,
                                             TKVs...>::type;
};

template <typename TVarTypeDict>
auto FillVarTypeDict(TVarTypeDict&& curDict)
{
    return std::forward<TVarTypeDict>(curDict);
}

template <typename TVarTypeDict, typename TCur, typename... TKVs>
auto FillVarTypeDict(TVarTypeDict&& curDict)
{
    auto newDict = std::forward<TVarTypeDict>(curDict).template Set<typename TCur::KeyType>(std::declval<typename TCur::ValueType>());
    return FillVarTypeDict<decltype(newDict), TKVs...>(std::move(newDict));
}

template <typename TVarTypeDict, typename SeqCont>
struct VarTypeDict2IOMap_;

template <typename TVarTypeDict, int... IDs>
struct VarTypeDict2IOMap_<TVarTypeDict, ContMetaFun::Helper::IndexSequence<IDs...>>
{
    using type = LayerIOMap<LayerKV<typename TVarTypeDict::template KeyType<IDs>,
                                    typename TVarTypeDict::template ValueType<IDs>
                                   >...>;
};
}

template <typename TLayer, typename TLayerIOMap>
struct LayerIOMapTrasfer_;

template <typename TLayer, typename... TKVs>
struct LayerIOMapTrasfer_<TLayer, LayerIOMap<TKVs...>>
{
    using TVarTypeDictOri = NSLayerIOMapTrasfer::CreateVarTypeDict_<VarTypeDict<>, TKVs...>;
    using RVarTypeDictOri = typename TVarTypeDictOri::type;
    
    using RVarTypeDictCre = decltype(RVarTypeDictOri::Create());
    using TVarTypeDictFill = decltype(NSLayerIOMapTrasfer::FillVarTypeDict<RVarTypeDictCre, TKVs...>(RVarTypeDictOri::Create()));
    
    using TForwardRes = decltype(std::declval<TLayer>().FeedForward(std::declval<TVarTypeDictFill>()));
    
    using IndexSeq = ContMetaFun::Helper::MakeIndexSequence<TForwardRes::Length>;
    
    using type = typename NSLayerIOMapTrasfer::VarTypeDict2IOMap_<TForwardRes, IndexSeq>::type;
};

template <typename TLayer>
using LayerInputMap = typename TLayer::InputTypeMap;

template <typename TLayer, typename TLayerIOMap>
using LayerOutputMap = typename LayerIOMapTrasfer_<TLayer, typename TLayer::InputTypeMap>::type;

template <typename TStoreType, bool store>
using LayerInternalBuf = std::conditional_t<store, std::stack<TStoreType>, NullParameter>;

template <bool IsAimDynamic, typename T>
auto DynamicTransWithFlag(T&& val)
{
    if constexpr (IsAimDynamic)
    {
        return MakeDynamic(std::forward<T>(val));
    }
    else
    {
        return std::forward<T>(val);
    }
}

namespace NSShapePromote
{
    template <typename T>
    constexpr size_t ShapeIndex = (size_t)-1;
    
    template <>
    constexpr size_t ShapeIndex<Shape<CategoryTags::Scalar>> = 0;
    
    template <>
    constexpr size_t ShapeIndex<Shape<CategoryTags::Matrix>> = 1;
    
    template <>
    constexpr size_t ShapeIndex<Shape<CategoryTags::ThreeDArray>> = 2;
    
    template <typename TSubCate>
    constexpr size_t ShapeIndex<Shape<CategoryTags::Batch<TSubCate>>> = 10 + ShapeIndex<Shape<TSubCate>>;
    
    template <typename TSubCate>
    constexpr size_t ShapeIndex<Shape<CategoryTags::Sequence<TSubCate>>> = 10 + ShapeIndex<Shape<TSubCate>>;
    
    template <typename TSubCate>
    constexpr size_t ShapeIndex<Shape<CategoryTags::BatchSequence<TSubCate>>> = 100 + ShapeIndex<Shape<TSubCate>>;
    
    template <typename TShape1, typename TShape2,
              typename = std::enable_if_t<(ShapeIndex<TShape1> > ShapeIndex<TShape2>)>>
    auto ShapePromoteHelper(const TShape1& shape1, const TShape2& shape2)
    {
        return ShapePromoteHelper(shape2, shape1);
    }
    
    template <typename TShape>
    auto ShapePromoteHelper(const Shape<CategoryTags::Scalar>&, const TShape& s2)
    {
        return s2;
    }
    
    template <typename TShape,
              typename = std::enable_if_t<(ShapeIndex<TShape> >= 1)>>
    auto ShapePromoteHelper(const Shape<CategoryTags::Matrix>& s1, const TShape& s2)
    {
        if ((s1.RowNum() != s2.RowNum()) || (s1.ColNum() != s2.ColNum()))
        {
            throw std::runtime_error("Shape promote error: shape mismatch.");
        }
        return s2;
    }
    
    template <typename TShape,
              typename = std::enable_if_t<(ShapeIndex<TShape> >= 2)>>
    auto ShapePromoteHelper(const Shape<CategoryTags::ThreeDArray>& s1, const TShape& s2)
    {
        if ((s1.RowNum() != s2.RowNum()) ||
            (s1.ColNum() != s2.ColNum()) ||
            (s1.PageNum() != s2.PageNum()))
        {
            throw std::runtime_error("Shape promote error: shape mismatch.");
        }
        return s2;
    }
}

template <typename TShape>
auto ShapePromote(const TShape& s)
{
    return s;
}

template <typename TShape1, typename TShape2, typename... TShapes>
auto ShapePromote(const TShape1& s1, const TShape2& s2, const TShapes&... rem)
{
    auto res = NSShapePromote::ShapePromoteHelper(s1, s2);
    return ShapePromote(res, rem...);
}
}