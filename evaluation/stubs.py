import sys, types
osgeo = types.ModuleType("osgeo")
for n in ("gdal", "ogr", "osr"):
    m = types.ModuleType(n); setattr(osgeo, n, m); sys.modules[f"osgeo.{n}"] = m
sys.modules["osgeo"] = osgeo
sys.modules.setdefault('fiona', types.ModuleType('fiona'))
