extends Node3D

# Camera
var CAMERA_BEGIN_POS: Vector3
var CAMERA_LOOK_AT: Vector3
const ROTATION_AXIS: Vector3 = Vector3.RIGHT
const CAMERA_INTEROP_PROGRESS_PER_SE := 0.5
var camera_begin_quat: Quaternion
var camera_end_quat: Quaternion
var pivot_camera_vec: Vector3:
	get:
		return CAMERA_BEGIN_POS - $RotationPoint.position

var progress: float = 0.0


# Called when the node enters the scene tree for the first time.
func _ready():
	setup_camera()


func _process(delta):
	if (Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_S)):
		var delta_progress = delta * CAMERA_INTEROP_PROGRESS_PER_SE
		if Input.is_key_pressed(KEY_S):
			delta_progress *= -1
		progress += delta_progress
		progress = clampf(progress, 0.0, 1.0)
		var interpolated_quat = camera_begin_quat.slerp(camera_end_quat, progress)
		
		var new_position = $RotationPoint.position \
			+ Vector3(interpolated_quat.x, interpolated_quat.y, interpolated_quat.z) \
			* pivot_camera_vec.length()
		
		$FlightCamera.position = new_position
		$FlightCamera.look_at(CAMERA_LOOK_AT + $FlightCamera.position)

func setup_camera():
	# TODO: to config
	$FlightCamera.transform = $FlightCamera.transform.rotated_local(Vector3.RIGHT, -65 * PI / 180)
	CAMERA_LOOK_AT = -$FlightCamera.transform.basis.z
	CAMERA_BEGIN_POS = $FlightCamera.position
	$FlightCamera.current = true
	
	# Computer camera end position
	camera_begin_quat = Quaternion(pivot_camera_vec.x, pivot_camera_vec.y, pivot_camera_vec.z, 0).normalized()
	var q = Quaternion(ROTATION_AXIS, PI / 2 - 1e-7)
	camera_end_quat = q * camera_begin_quat * q.inverse()
