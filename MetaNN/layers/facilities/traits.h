#pragma once

#include <stack>
#include <type_traits>
#include <MetaNN/facilities/var_type_dict.h>
#include <MetaNN/layers/facilities/layer_io_map.h>

namespace MetaNN::LayerTraits
{
template <typename TWeight, typename TGradStack, typename TGradCollector>
void MatrixGradCollect(const TWeight& weight,
                       TGradStack& gradStack,
                       TGradCollector& col)
{
    size_t stackSize = gradStack.size();
    if (stackSize == 0) return;
    
    if (stackSize == 1)
    {
        auto g = gradStack.top();
        gradStack.pop();
        col.Collect(weight, MakeDynamic(std::move(g)));
        return;
    }
    else
    {
        DynamicBatch<RemConstRef<typename TGradStack::value_type>> dBatch(weight.Shape());
        while (!gradStack.empty())
        {
            auto g = gradStack.top();
            gradStack.pop();
            dBatch.PushBack(std::move(g));
        }
        auto tmp = Collapse(dBatch, weight.Shape());
        col.Collect(weight, MakeDynamic(std::move(tmp)));
        return;
    }
}

template <typename TTypeMap, typename TKey, typename TCont>
auto PickItemFromCont(TCont&& cont)
{
    using TAim = typename TTypeMap::template Find<TKey>;
    auto itemOri = std::forward<TCont>(cont).template Get<TKey>();
    static_assert(!std::is_same_v<decltype(itemOri), NullParameter>);
    static_assert(!std::is_same_v<TAim, NullParameter>);
    
    if constexpr (IsDynamic<TAim>)
    {
        auto res = MakeDynamic(std::move(itemOri));
        static_assert(std::is_same_v<decltype(res), TAim>);
        return res;
    }
    else
    {
        static_assert(std::is_same_v<RemConstRef<decltype(itemOri)>, TAim>);
        return itemOri;
    }
}

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

template <typename TLayer, typename TLayerIOMap>
struct LayerIOMapForwardTrasfer_;

template <typename TLayer, typename... TKVs>
struct LayerIOMapForwardTrasfer_<TLayer, LayerIOMap<TKVs...>>
{
    using TVarTypeDictOri = CreateVarTypeDict_<VarTypeDict<>, TKVs...>;
    using RVarTypeDictOri = typename TVarTypeDictOri::type;
    
    using RVarTypeDictCre = decltype(RVarTypeDictOri::Create());
    using TVarTypeDictFill = decltype(FillVarTypeDict<RVarTypeDictCre, TKVs...>(RVarTypeDictOri::Create()));
    
    using TForwardRes = decltype(std::declval<TLayer>().FeedForward(std::declval<TVarTypeDictFill>()));
    
    using IndexSeq = ContMetaFun::Helper::MakeIndexSequence<TForwardRes::Length>;
    
    using type = typename VarTypeDict2IOMap_<TForwardRes, IndexSeq>::type;
};

template <typename TLayer, typename TLayerIOMap>
struct LayerIOMapBackwardTrasfer_;

template <typename TLayer, typename... TKVs>
struct LayerIOMapBackwardTrasfer_<TLayer, LayerIOMap<TKVs...>>
{
    using TVarTypeDictOri = CreateVarTypeDict_<VarTypeDict<>, TKVs...>;
    using RVarTypeDictOri = typename TVarTypeDictOri::type;
    
    using RVarTypeDictCre = decltype(RVarTypeDictOri::Create());
    using TVarTypeDictFill = decltype(FillVarTypeDict<RVarTypeDictCre, TKVs...>(RVarTypeDictOri::Create()));
    
    using TForwardRes = decltype(std::declval<TLayer>().FeedBackward(std::declval<TVarTypeDictFill>()));
    
    using IndexSeq = ContMetaFun::Helper::MakeIndexSequence<TForwardRes::Length>;
    
    using type = typename VarTypeDict2IOMap_<TForwardRes, IndexSeq>::type;
};
}

template <typename TLayer>
using LayerInputItemTypes = typename TLayer::InputItemTypes;

template <typename TLayer, typename TLayerIOMap>
using LayerOutputItemTypes = typename NSLayerIOMapTrasfer::LayerIOMapForwardTrasfer_<TLayer, typename TLayer::InputItemTypes>::type;

template <typename TLayer>
using LayerInputGradTypes = typename TLayer::InputGradTypes;

template <typename TLayer, typename TLayerIOMap>
using LayerOutputGradTypes = typename NSLayerIOMapTrasfer::LayerIOMapBackwardTrasfer_<TLayer, typename TLayer::InputGradTypes>::type;

template <typename TStoreType, bool store>
using LayerInternalBuf = std::conditional_t<store, std::stack<TStoreType>, NullParameter>;

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
    
    template <typename TShape1, typename TShape2>
    auto ShapePromoteHelper(const TShape1& shape1, const TShape2& shape2)
    {
        if constexpr (ShapeIndex<TShape1> > ShapeIndex<TShape2>)
        {
            return ShapePromoteHelper(shape2, shape1);
        }
        else if constexpr (ShapeIndex<TShape1> == ShapeIndex<TShape2>)
        {
            if (shape1 != shape2)
            {
                throw std::runtime_error("Shape promote error: shape mismatch.");
            }
            return shape1;
        }
        else if constexpr ((std::is_same_v<TShape1, Shape<CategoryTags::Scalar>>) &&
                           (ShapeIndex<TShape2> >= 0))
        {
            return shape2;
        }
        else if constexpr ((std::is_same_v<TShape1, Shape<CategoryTags::Matrix>>) &&
                           (ShapeIndex<TShape2> >= 1))
        {
            if ((shape1.RowNum() != shape2.RowNum()) || (shape1.ColNum() != shape2.ColNum()))
            {
                throw std::runtime_error("Shape promote error: shape mismatch.");
            }
            return shape2;
        }
        else if constexpr ((std::is_same_v<TShape1, Shape<CategoryTags::ThreeDArray>>) &&
                           (ShapeIndex<TShape2> >= 2))
        {
            if ((shape1.RowNum() != shape2.RowNum()) ||
                (shape1.ColNum() != shape2.ColNum()) ||
                (shape1.PageNum() != shape2.PageNum()))
            {
                throw std::runtime_error("Shape promote error: shape mismatch.");
            }
            return shape2;
        }
        else
        {
            static_assert(DependencyFalse<TShape1>);
        }
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