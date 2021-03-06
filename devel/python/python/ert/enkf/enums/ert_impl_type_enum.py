#  Copyright (C) 2013  Statoil ASA, Norway.
#
#  The file 'ert_impl_type_enum.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
from ert.cwrap import BaseCEnum
from ert.enkf import ENKF_LIB


class ErtImplType(BaseCEnum):
    INVALID          = None
    IMPL_TYPE_OFFSET = None
    STATIC           = None       # MULTZ has been removed & MULTFLT
    FIELD            = None       # WELL has been removed
    GEN_KW           = None       # RELPERM has been removed & HAVANA_FAULT
    SUMMARY          = None       # TPGZONE has been removed
    GEN_DATA         = None       # PILOT_POINT has been removed
    SURFACE          = None
    CONTAINER        = None


ErtImplType.addEnum("INVALID", 0)
ErtImplType.addEnum("IMPL_TYPE_OFFSET", 100)
ErtImplType.addEnum("STATIC", 100)
ErtImplType.addEnum("FIELD", 104)
ErtImplType.addEnum("GEN_KW", 107)
ErtImplType.addEnum("SUMMARY", 110)
ErtImplType.addEnum("GEN_DATA", 113)
ErtImplType.addEnum("SURFACE", 114)
ErtImplType.addEnum("CONTAINER", 115)
ErtImplType.registerEnum(ENKF_LIB, "ert_impl_type_enum")



