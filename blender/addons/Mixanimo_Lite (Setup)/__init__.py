# ===============================================================================
# Copyright (c) 2025 Mixanimo
# All rights reserved.
# This addon is provided under the MIT License.
# See LICENSE file for full license text.
# ===============================================================================

bl_info = {
    "name": "Mixanimo Lite",
    "author": "Mixanimo",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "3D View > Sidebar > Mixanimo Addon",
    "description": (
        "Mixamo actions: reorder, repeat, overlap."
        "Bake and advanced features available in Pro version."
    ),
    "category": "Animation"
}

import bpy
from mathutils import Vector
from bpy.types import (
    Panel,
    Operator,
    PropertyGroup,
    UIList
)
from bpy.props import (
    StringProperty,
    CollectionProperty,
    IntProperty,
    BoolProperty
)
from bpy.app.handlers import persistent

#####################################################
# Helper Functions (Local Space)
#####################################################

def get_local_location_at_frame(action, bone_name, frame):
    loc = [0.0, 0.0, 0.0]
    data_path = f'pose.bones["{bone_name}"].location'
    for fc in action.fcurves:
        if fc.data_path == data_path:
            idx = fc.array_index
            loc[idx] = fc.evaluate(frame)
    return Vector(loc)

def get_action_start_location_local(action, bone_name):
    start_frame = int(action.frame_range[0])
    return get_local_location_at_frame(action, bone_name, start_frame)

def get_action_end_location_local(action, bone_name):
    end_frame = int(action.frame_range[1])
    return get_local_location_at_frame(action, bone_name, end_frame)

def offset_action_root_local(action, bone_name, offset_vec):
    data_path = f'pose.bones["{bone_name}"].location'
    fcurves = [fc for fc in action.fcurves if fc.data_path == data_path]
    if fcurves:
        for i, fc in enumerate(fcurves):
            for kp in fc.keyframe_points:
                if i < len(offset_vec):
                    kp.co[1] += offset_vec[i]
            fc.update()
    else:
        print(f"Warning: No fcurves found for bone '{bone_name}' in action '{action.name}'.")

def create_action_copy(original_action, repeat_index=1):
    new_action = original_action.copy()
    new_action.name = f"{original_action.name}_REPEAT_{repeat_index}"
    return new_action

#####################################################
# 1) SEND TO TIMELINE
#####################################################

def send_to_timeline(armature, actions, root_bone="mixamorig:Hips", overlap_frames=3):
    if armature.animation_data:
        armature.animation_data_clear()
    armature.animation_data_create()
    
    prev_end_loc = Vector((0, 0, 0))
    current_start = 1
    track_index = 1
    
    for (act, repeats) in actions:
        for r in range(repeats):
            new_act = create_action_copy(act, repeat_index=r+1)
            length = int(new_act.frame_range[1] - new_act.frame_range[0])
            
            track = armature.animation_data.nla_tracks.new()
            track.name = f"Track_Send_{track_index}_{new_act.name}"
            track_index += 1
            
            if track_index == 2:
                strip_start = current_start
            else:
                strip_start = current_start - overlap_frames
                if strip_start < 1:
                    strip_start = 1
            strip_end = strip_start + length
            
            strip_start_int = int(round(strip_start))
            strip_end_int = int(round(strip_end))
            
            strip = track.strips.new(new_act.name, strip_start_int, new_act)
            strip.frame_start = strip_start_int
            strip.frame_end   = strip_end_int
            
            if track_index != 2:
                strip.blend_in   = overlap_frames
                strip.blend_type = 'REPLACE'
            
            start_loc = get_action_start_location_local(new_act, root_bone)
            if track_index > 2:
                offset = prev_end_loc - start_loc
                offset_action_root_local(new_act, root_bone, offset)
            else:
                offset = Vector((0, 0, 0))
            
            prev_end_loc = get_action_end_location_local(new_act, root_bone)
            current_start = strip_end_int + 1

#####################################################
# PROPERTY GROUP & UIList
#####################################################

def action_index_update(self, context):
    if self.action_index < 0 or self.action_index >= len(self.action_collection):
        return
    item = self.action_collection[self.action_index]
    arm_name = item.armature_name
    if arm_name and arm_name in bpy.data.objects:
        arm_obj = bpy.data.objects[arm_name]
        bpy.ops.object.select_all(action='DESELECT')
        arm_obj.select_set(True)
        context.view_layer.objects.active = arm_obj

class MixamoActionItem(PropertyGroup):
    action_name: StringProperty()
    armature_name: StringProperty(default="")
    use_action: BoolProperty(
        name="",
        description="Enable Action",
        default=True
    )
    repeat_count: IntProperty(
        name="Count",
        description=" Repeat Count",
        default=1,
        min=1,
        max=999
    )

class MixamoUnifiedSettings(PropertyGroup):
    root_bone: StringProperty(
        name="Root Bone",
        description="Name of the root bone (usually 'mixamorig:Hips' in Mixamo)",
        default="mixamorig:Hips"
    )
    overlap_frames: IntProperty(
        name="Overlap Frames",
        description="Number of frames used for blending between strips",
        default=3,
        min=0,
        max=100
    )
    action_collection: CollectionProperty(type=MixamoActionItem)
    action_index: IntProperty(
        default=0,
        update=action_index_update
    )
    order_confirmed: BoolProperty(default=False)

class MIXAMO_UL_ActionList(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, "use_action", text="")
            row.prop(item, "action_name", text="", emboss=False, icon='ACTION')
            if item.armature_name:
                row.label(text=f"[{item.armature_name}]", icon='ARMATURE_DATA')
            act = bpy.data.actions.get(item.action_name)
            if act:
                row.label(text=f"{int(act.frame_range[1] - act.frame_range[0])} fr", icon='TIME')
            row.prop(item, "repeat_count", text="x")
        elif self.layout_type == 'GRID':
            pass

#####################################################
# OPERATORS (Lite Only)
#####################################################

class MIXAMO_OT_GetActions(Operator):
    bl_idname = "mixamo.get_actions"
    bl_label = "Load Actions"
    bl_description = "List actions in the scene"
    
    def execute(self, context):
        prefs = context.scene.mixamo_unified
        prefs.action_collection.clear()
        
        for act in bpy.data.actions:
            if "_REPEAT_" not in act.name:
                for fc in act.fcurves:
                    if "pose.bones[" in fc.data_path:
                        item = prefs.action_collection.add()
                        item.action_name = act.name
                        for obj in bpy.data.objects:
                            if obj.type == 'ARMATURE' and obj.animation_data:
                                if obj.animation_data.action == act:
                                    item.armature_name = obj.name
                                    break
                        break
        
        prefs.order_confirmed = False
        self.report({'INFO'}, "Actions loaded. (Armature references assigned if found.)")
        return {'FINISHED'}

class MIXAMO_OT_MoveActionUp(Operator):
    bl_idname = "mixamo.move_action_up"
    bl_label = "Move Action Up"
    
    def execute(self, context):
        prefs = context.scene.mixamo_unified
        idx = prefs.action_index
        if idx > 0:
            prefs.action_collection.move(idx, idx-1)
            prefs.action_index -= 1
            prefs.order_confirmed = False
        return {'FINISHED'}

class MIXAMO_OT_MoveActionDown(Operator):
    bl_idname = "mixamo.move_action_down"
    bl_label = "Move Action Down"
    
    def execute(self, context):
        prefs = context.scene.mixamo_unified
        idx = prefs.action_index
        if idx < len(prefs.action_collection) - 1:
            prefs.action_collection.move(idx, idx+1)
            prefs.action_index += 1
            prefs.order_confirmed = False
        return {'FINISHED'}

class MIXAMO_OT_ConfirmOrder(Operator):
    bl_idname = "mixamo.confirm_order"
    bl_label = "Confirm Order"
    bl_description = "Assign selected actions to active Armature in specified order."
    
    def execute(self, context):
        prefs = context.scene.mixamo_unified
        if len(prefs.action_collection) == 0:
            self.report({'WARNING'}, "No action found.")
            return {'CANCELLED'}
        
        first_item = prefs.action_collection[0]
        if first_item.armature_name and first_item.armature_name in bpy.data.objects:
            arm = bpy.data.objects[first_item.armature_name]
            bpy.context.view_layer.objects.active = arm
            arm.select_set(True)
        else:
            self.report({'WARNING'}, "No armature assigned to the first action")
            return {'CANCELLED'}
        
        prefs.order_confirmed = True
        
        found_any = False
        for item in prefs.action_collection:
            if item.use_action:
                act = bpy.data.actions.get(item.action_name)
                if act:
                    found_any = True
        
        if found_any:
            self.report({'INFO'}, "Order confirmed. Actions ready.")
        else:
            self.report({'INFO'}, "No action selected (use_action).")
        
        return {'FINISHED'}

class MIXAMO_OT_SendToTimeline(Operator):
    bl_idname = "mixamo.send_to_timeline"
    bl_label = "Send to Timeline"
    bl_description = "Add selected actions to NLA editor in overlapped sequence."
    
    def execute(self, context):
        prefs = context.scene.mixamo_unified
        if not prefs.order_confirmed:
            self.report({'WARNING'}, "Run 'Confirm Order' before proceeding")
            return {'CANCELLED'}
        
        arm = context.active_object
        if not arm or arm.type != 'ARMATURE':
            self.report({'WARNING'}, "Please select an armature")
            return {'CANCELLED'}
        
        actions_to_send = []
        for item in prefs.action_collection:
            if item.use_action:
                act = bpy.data.actions.get(item.action_name)
                if act:
                    actions_to_send.append((act, item.repeat_count))
        
        if not actions_to_send:
            self.report({'WARNING'}, "No action selected (use_action=True). Operation cancelled")
            return {'CANCELLED'}
        
        send_to_timeline(arm, actions_to_send, root_bone=prefs.root_bone, overlap_frames=prefs.overlap_frames)
        
        self.report({'INFO'}, f"{len(actions_to_send)} action entry(ies) sent to timeline (multi-track).")
        return {'FINISHED'}

@persistent
def sync_selection_from_armature(scene):
    prefs = bpy.context.scene.mixamo_unified
    arm = bpy.context.view_layer.objects.active
    
    if not arm or arm.type != 'ARMATURE':
        return
    if not arm.animation_data or not arm.animation_data.action:
        return
    
    current_action = arm.animation_data.action
    for i, item in enumerate(prefs.action_collection):
        if item.action_name == current_action.name:
            if prefs.action_index != i:
                prefs.action_index = i
            break

class MIXAMO_PT_UnifiedPanel(Panel):
    bl_label = "MIXANIMO Lite"
    bl_idname = "MIXAMO_PT_unified_merger"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Mixanimo Tools"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Load & Organize Animations", icon="ANIM")
        layout.separator()
        prefs = context.scene.mixamo_unified
        
        row = layout.row()
        row.operator("mixamo.get_actions", text="Load Actions", icon='FILE_REFRESH')
        
        row = layout.row()
        row.template_list(
            "MIXAMO_UL_ActionList",
            "",
            prefs,
            "action_collection",
            prefs,
            "action_index",
            rows=5
        )
        col = row.column(align=True)
        col.operator("mixamo.move_action_up", icon='TRIA_UP', text="")
        col.operator("mixamo.move_action_down", icon='TRIA_DOWN', text="")
        
        box = layout.box()
        box.prop(prefs, "root_bone", text="Root Bone")
        box.prop(prefs, "overlap_frames", text="Overlap Frames")
        box.operator("mixamo.confirm_order", text="Confirm Order", icon='CHECKMARK')
        
        layout.separator()
        layout.label(text="Nonlinear Animation", icon="NLA")
        layout.separator()
        row = layout.row(align=True)
        row.operator("mixamo.send_to_timeline", text="Send", icon='NLA')
        
        # Kilitli, pasif butonlar:
        row = layout.row(align=True)
        row.enabled = False
        row.operator("wm.locked_bake", text="Bake Action (LOCKED)", icon='LOCKED')
        row.operator("wm.locked_stabilize", text="Stabilize Root (LOCKED)", icon='LOCKED')

        layout.separator()
        box = layout.box()
        box.label(text="Unlock these features in Mixanimo Pro!", icon="INFO")
        box.operator("wm.url_open", text="Check out Pro", icon="URL").url = "https://mixanimo.gumroad.com"

# LOCKED operatorları (çalışmayan dummy):
class WM_OT_LockedBake(Operator):
    bl_idname = "wm.locked_bake"
    bl_label = "Bake Action (LOCKED)"
    def execute(self, context):
        self.report({'INFO'}, "Bu özellik sadece Pro sürümde mevcuttur.")
        return {'CANCELLED'}

class WM_OT_LockedStabilize(Operator):
    bl_idname = "wm.locked_stabilize"
    bl_label = "Stabilize Root (LOCKED)"
    def execute(self, context):
        self.report({'INFO'}, "Bu özellik sadece Pro sürümde mevcuttur.")
        return {'CANCELLED'}

classes = (
    MixamoActionItem,
    MixamoUnifiedSettings,
    MIXAMO_UL_ActionList,
    MIXAMO_OT_GetActions,
    MIXAMO_OT_MoveActionUp,
    MIXAMO_OT_MoveActionDown,
    MIXAMO_OT_ConfirmOrder,
    MIXAMO_OT_SendToTimeline,
    MIXAMO_PT_UnifiedPanel,
    WM_OT_LockedBake,
    WM_OT_LockedStabilize,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mixamo_unified = bpy.props.PointerProperty(type=MixamoUnifiedSettings)
    
    if sync_selection_from_armature not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(sync_selection_from_armature)

def unregister():
    if sync_selection_from_armature in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(sync_selection_from_armature)
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.mixamo_unified

if __name__ == "__main__":
    register()