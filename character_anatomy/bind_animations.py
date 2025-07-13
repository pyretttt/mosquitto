import bpy 

rig = bpy.context.object

for action in bpy.data.actions:
    rig.animation_data.action = action
    print(f"Bound animation: {action.name}"