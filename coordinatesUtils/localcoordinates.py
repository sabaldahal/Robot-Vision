###
#This script is meant to be run in blender environment where bpy library is present
###


import bpy
import json
import os
import re

col = bpy.data.collections.get('Keypoints')
obj = bpy.data.objects.get('scobj')

coords = []
for k in col.objects:
    coords.append({
        'name': k.name,
        'location': k.location[:]
    })


def tryThisToo():
    coords = []
    for k in col.objects:
        local_matrix = obj.matrix_world.inverted() @ k.matrix_world
        local_location = local_matrix.to_translation()
        coords.append({
            'name': k.name,
            'location': local_location[:]
        })    

ks = sorted(
    coords,
    key=lambda d: (
        re.sub(r'_\d+$', '', d['name']),
        int(re.search(r'(\d+)$', d['name']).group())
    )
)


wd = os.getcwd()
file = os.path.join(wd, "coords.json")
with open(file, "w") as f:
    json.dump(ks, f, indent=4)
    print(f"file saved to: {file}")
