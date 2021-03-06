#  Copyright (C) 2012  Statoil ASA, Norway. 
#   
#  The file 'enkf_fs.py' is part of ERT - Ensemble based Reservoir Tool. 
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
from ert.cwrap import BaseCClass, CWrapper
from ert.enkf import ENKF_LIB, TimeMap, StateMap
from ert.enkf.enums import EnKFFSType


class EnkfFs(BaseCClass):
    def __init__(self, mount_point):
        c_ptr = EnkfFs.cNamespace().mount(mount_point)
        super(EnkfFs, self).__init__(c_ptr)

        self.__umounted = False # Keep track of umounting so we only do it once


    @classmethod
    def createCReference(cls, c_pointer, parent=None):
        obj = super(EnkfFs, cls).createCReference(c_pointer, parent)
        obj.__umounted = False
        return obj


    # def has_node(self, node_key, var_type, report_step, iens, state):
    #     return EnkfFs.cNamespace().has_node(self, node_key, var_type, report_step, iens, state)
    #
    # def has_vector(self, node_key, var_type, iens, state):
    #     return EnkfFs.cNamespace().has_vector(self, node_key, var_type, iens, state)
    #
    #
    # def fread_node(self, key, type, step, member, value):
    #     buffer = Buffer(100)
    #     EnkfFs.cNamespace().fread_node(self, buffer, key, type, step, member, value)
    #
    # def fread_vector(self, key, type, member, value):
    #     buffer = Buffer(100)
    #     EnkfFs.cNamespace().fread_vector(self, buffer, key, type, member, value)

    def getTimeMap(self):
        """ @rtype: TimeMap """
        self.__checkIfUmounted()
        return EnkfFs.cNamespace().get_time_map(self).setParent(self)

    def getStateMap(self):
        """ @rtype: StateMap """
        self.__checkIfUmounted()
        return EnkfFs.cNamespace().get_state_map(self).setParent(self)

    def getCaseName(self):
        """ @rtype: str """
        self.__checkIfUmounted()
        return EnkfFs.cNamespace().get_case_name(self)

    def isReadOnly(self):
        """ @rtype: bool """
        self.__checkIfUmounted()
        return EnkfFs.cNamespace().is_read_only(self)


    def refCount(self):
        self.__checkIfUmounted()
        return self.cNamespace().get_refcount(self)


    @classmethod
    def exists(cls, path):
        return cls.cNamespace().exists(path)

    @classmethod
    def createFileSystem(cls, path, fs_type, arg=None):
        assert isinstance(path, str)
        assert isinstance(fs_type, EnKFFSType)
        cls.cNamespace().create(path, fs_type, arg)


    def __checkIfUmounted(self):
        if self.__umounted:
            raise AssertionError("The EnkfFs instance has been umounted!")

    def umount(self):
        if not self.__umounted:
            EnkfFs.cNamespace().decref(self)
            self.__umounted = True

    def free(self):
        self.umount()


cwrapper = CWrapper(ENKF_LIB)
cwrapper.registerType("enkf_fs", EnkfFs)
cwrapper.registerType("enkf_fs_obj", EnkfFs.createPythonObject)
cwrapper.registerType("enkf_fs_ref", EnkfFs.createCReference)

EnkfFs.cNamespace().mount = cwrapper.prototype("c_void_p enkf_fs_mount(char* )")
EnkfFs.cNamespace().create = cwrapper.prototype("void enkf_fs_create_fs(char* , enkf_fs_type_enum , c_void_p)")
EnkfFs.cNamespace().decref = cwrapper.prototype("int enkf_fs_decref(enkf_fs)")
EnkfFs.cNamespace().get_refcount = cwrapper.prototype("int enkf_fs_get_refcount(enkf_fs)")
EnkfFs.cNamespace().has_node = cwrapper.prototype("bool enkf_fs_has_node(enkf_fs, char*, c_uint, int, int, c_uint)")
EnkfFs.cNamespace().has_vector = cwrapper.prototype("bool enkf_fs_has_vector(enkf_fs, char*, c_uint, int, c_uint)")
EnkfFs.cNamespace().fread_node = cwrapper.prototype("void enkf_fs_fread_node(enkf_fs, buffer, char*, c_uint, int, int, c_uint)")
EnkfFs.cNamespace().fread_vector = cwrapper.prototype("void enkf_fs_fread_vector(enkf_fs, buffer, char*, c_uint, int, c_uint)")
EnkfFs.cNamespace().get_time_map = cwrapper.prototype("time_map_ref enkf_fs_get_time_map(enkf_fs)")
EnkfFs.cNamespace().get_state_map = cwrapper.prototype("state_map_ref enkf_fs_get_state_map(enkf_fs)")
EnkfFs.cNamespace().exists = cwrapper.prototype("bool enkf_fs_exists(char*)")
EnkfFs.cNamespace().get_case_name = cwrapper.prototype("char* enkf_fs_get_case_name(enkf_fs)")
EnkfFs.cNamespace().is_read_only = cwrapper.prototype("bool enkf_fs_is_read_only(enkf_fs)")
