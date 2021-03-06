#  Copyright (C) 2011  Statoil ASA, Norway.
#
#  The file 'export_keyword_model.py' is part of ERT - Ensemble based Reservoir Tool.
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
from ert.enkf import EnkfVarType, ErtImplType
from ert_gui.models import ErtConnector

class ExportKeywordModel(ErtConnector):

    def __init__(self):
        super(ExportKeywordModel, self).__init__()
        self.__gen_kw = None
        self.__field_kw = None

    def getKeylistFromImplType(self, ert_impl_type):
        return sorted(self.ert().ensembleConfig().getKeylistFromImplType(ert_impl_type))


    def isDynamicField(self, key):
        config_node = self.ert().ensembleConfig().getNode(key)
        variable_type = config_node.getVariableType()
        return variable_type == EnkfVarType.DYNAMIC_STATE

    def getImplementationType(self, key):
        config_node = self.ert().ensembleConfig().getNode(key)
        return config_node.getImplementationType()

    def getGenKwKeyWords(self):
        if self.__gen_kw is None:
            self.__gen_kw = [key for key in self.ert().ensembleConfig().getKeylistFromImplType(ErtImplType.GEN_KW)]

        return self.__gen_kw

    def getFieldKeyWords(self):
        if self.__field_kw is None:
            self.__field_kw = self.getKeylistFromImplType(ErtImplType.FIELD)

        return self.__field_kw

    def getKeyWords(self):
        return sorted(self.getFieldKeyWords() + self.getGenKwKeyWords())

    def isGenKw(self, key):
        return key in self.__gen_kw

    def isFieldKw(self, key):
        return key in self.__field_kw