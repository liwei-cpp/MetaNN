#pragma once

#include <MetaNN/facilities/var_type_dict.h>

namespace MetaNN::ConvParams
{
    struct PadModeTypeCate
    {
        struct Default;
        struct Same;
    };
    
    struct RowNum;
    struct ColNum;
    struct PageNum;
}