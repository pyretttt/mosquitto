extends Node3D

# Camera
var CAMERA_BEGIN_POS: Vector3
var CAMERA_LOOK_AT: Vector3
var ROTATION_AXIS: Vector3 = Vector3(1, 0, 0).normalized()

var progress: float = 0.0
var camera_begin_quat: Quaternion
var camera_end_quat: Quaternion

var pivot_camera_vec: Vector3:
	get:
		return CAMERA_BEGIN_POS - $RotationPoint.position

# Called when the node enters the scene tree for the first time.
func _ready():
	setup_camera()


func _process(delta):
	progress = clampf(progress + delta / 5, 0, 1)
	var interpolated = camera_begin_quat.slerp(camera_end_quat, progress)
	var new_position = $RotationPoint.position \
		+ Vector3(interpolated.x, interpolated.y, interpolated.z) \
		* pivot_camera_vec.length()
	var up = (-CAMERA_LOOK_AT).cross($FlightCamera.transform.basis.x).normalized()
	$FlightCamera.look_at_from_position(new_position, -CAMERA_LOOK_AT, up)
	
	
	print($FlightCamera.transform)

func setup_camera():
	# TODO: to config
	CAMERA_LOOK_AT = -$FlightCamera/Camera3D.transform.basis.z
	CAMERA_BEGIN_POS = $FlightCamera.position
	#$FlightCamera/Camera3D.make_current()
	
	# Computer camera end position
	var pivot_camera_direction = pivot_camera_vec.normalized()
	camera_begin_quat = Quaternion(pivot_camera_direction.x, pivot_camera_direction.y, pivot_camera_direction.z, 0)
	var q = Quaternion(ROTATION_AXIS, PI / 2)
	camera_end_quat = q * camera_begin_quat * q.inverse()
