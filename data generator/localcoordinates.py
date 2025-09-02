###
#This script is meant to be run in blender environment where bpy library is present
###


import bpy
import json
import os

col = bpy.data.collections.get('Keypoints')
obj = bpy.data.objects.get('scobj')

coords = []
for k in col.objects:
    coords.append({
        'name': k.name,
        'location': k.location[:]
    })

ks = sorted(coords, key=lambda kp: kp["name"])

wd = os.getcwd()
file = os.path.join(wd, "coords.json")
with open(file, "w") as f:
    json.dump(ks, f, indent=4)
    print(f"file saved to: {file}")
