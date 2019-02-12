#pragma once

#include <MetaNN/facilities/var_type_dict.h>

namespace MetaNN
{
struct CostLayerIn : public VarTypeDict<CostLayerIn, struct CostLayerLabel> {};

struct RnnLayerHiddenBefore;
}
