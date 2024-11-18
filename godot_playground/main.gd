extends Node3D

const CAMERA_POS = Vector3(21, 43, 15)
const CAMERA_ROTATION = Vector3(-67.5, 29, 0)

var rotation_axis: Vector3 = Vector3(7, 0, -7).normalized()
var new_camera_position: Quaternion

var progress: float = 0.0
var p: Quaternion

# Called when the node enters the scene tree for the first time.
func _ready():
	$FlightCamera.set_identity()
	# TODO: to config
	$FlightCamera.position = CAMERA_POS
	$FlightCamera.rotation_degrees = CAMERA_ROTATION
	$FlightCamera/Camera3D.make_current()
	# camera
	var pivot_pos = $RotationPoint.position
	var camera_position: Vector3 = CAMERA_POS - pivot_pos
	var rotation_vector = camera_position.normalized()
	p = Quaternion(rotation_vector.x, rotation_vector.y, rotation_vector.z, 0)
	var q = Quaternion(rotation_axis, PI / 2)
	var q_inv = q.inverse()
	var rotated_vector = q * p * q_inv
	
	#var new_camera_position = camera_position.length() * Vector3(rotated_vector.x, rotated_vector.y, rotated_vector.z)
	new_camera_position = rotated_vector

func _process(delta):
	pass
	progress = clampf(progress + delta / 5, 0, 1)
	var interpolated = p.slerp(new_camera_position, progress)
	var pivot_pos = $RotationPoint.position
	var camera_position: Vector3 = CAMERA_POS - pivot_pos
	print("Camera position: ", camera_position)
	var pos = pivot_pos + camera_position.length() * Vector3(interpolated.x, interpolated.y, interpolated.z)
	$FlightCamera.position = pos
