import os
from .polyline import Polyline

class XYZReader(object):

    @staticmethod
    def readXYZFile(path):
        """ @rtype: Polyline """

        if not os.path.exists(path):
            raise ValueError("Path does not exist '%s'!" % path)

        name = os.path.basename(path)

        polyline = Polyline(name=name)

        with open(path, "r") as f:
            for line in f:
                x, y, z = map(float, line.split())

                if x != 999.000000 and y != 999.000000 and z != 999.000000:
                    polyline.addPoint(x, y, z)

        return polyline
