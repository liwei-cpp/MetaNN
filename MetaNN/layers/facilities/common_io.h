#pragma once

#include <MetaNN/facilities/var_type_dict.h>

namespace MetaNN
{
struct LayerIO : public VarTypeDict<LayerIO> {};

struct CostLayerIn : public VarTypeDict<CostLayerIn, struct CostLayerLabel> {};

struct RnnLayerHiddenBefore;
}
