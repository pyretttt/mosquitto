import bpy

armature = bpy.context.object
if armature and armature.type == "ARMATURE":
    for bone in armature.data.bones:
        if bone.name.startswith("ORG"):
            bone.name = bone.name.replace("ORG", "DEF", 1)
            print("Renamed bone.name")
            
else:
    print("Select an armature first")