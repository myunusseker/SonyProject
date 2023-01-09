import os
import sys
from object2urdf import ObjectUrdfBuilder

# Build single URDFs

builder = ObjectUrdfBuilder('./', urdf_prototype='objects/_prototype.urdf')
builder.build_urdf(filename="objects/plate.obj", force_overwrite=True, decompose_concave=True, force_decompose=False, center='mass')
