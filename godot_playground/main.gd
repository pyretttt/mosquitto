extends Node3D

# Camera
const CAMERA_ANGLE_BEGIN = PI / 2 - 1e-8
const CAMERA_ELLIPSE_AB = Vector2(48, 27)

var CAMERA_BEGIN_POS: Vector3
var CAMERA_LOOK_AT: Vector3
const ROTATION_AXIS: Vector3 = Vector3.FORWARD
const CAMERA_INTEROP_PROGRESS_PER_SE := 0.5
var camera_begin_quat: Quaternion
var camera_end_quat: Quaternion
var pivot_camera_vec: Vector3:
	get:
		return CAMERA_BEGIN_POS - $RotationPoint.position

# State
var progress: float = 0.0
var cameraMovingDirection := 0

# Called when the node enters the scene tree for the first time.
func _ready():
	setup_camera()


func _process(delta):
	if (cameraMovingDirection != 0):
		var delta_progress = cameraMovingDirection * delta * CAMERA_INTEROP_PROGRESS_PER_SE
		progress += delta_progress
		progress = clampf(progress, 0.0, 1.0)
		var interpolated_quat = camera_begin_quat.slerp(camera_end_quat, progress)
		
		var new_position = $RotationPoint.position \
			+ Vector3(interpolated_quat.x, interpolated_quat.y, interpolated_quat.z) \
			* pivot_camera_vec.length()
		
		$FlightCamera.position = new_position
		$FlightCamera.look_at(CAMERA_LOOK_AT + new_position)
	

func setup_camera():
	# TODO: to config
	CAMERA_LOOK_AT = -$FlightCamera.transform.basis.z
	CAMERA_BEGIN_POS = $FlightCamera.position
	
	var camera_pos = camera_point(
		CAMERA_ANGLE_BEGIN, 
		CAMERA_ELLIPSE_AB.x,
		CAMERA_ELLIPSE_AB.y
	)
	$FlightCamera.transform.origin.x = camera_pos.x
	$FlightCamera.transform.origin.z = camera_pos.y
	
	# Computer camera end position
	#camera_begin_quat = Quaternion(pivot_camera_vec.x, pivot_camera_vec.y, pivot_camera_vec.z, 0).normalized()
	#var q = Quaternion(ROTATION_AXIS, PI / 2 - 1e-7)
	#camera_end_quat = q * camera_begin_quat * q.inverse()
	
func _input(event):
	if event is InputEventKey and event.pressed and (event.keycode == KEY_W or event.keycode == KEY_S):
		cameraMovingDirection = 1 if event.keycode == KEY_W else -1
	elif event is InputEventMagnifyGesture:
		print(event)
		cameraMovingDirection = event.factor - 1
	else:
		cameraMovingDirection = 0
		
		
func camera_point(angle, a, b):
	var sign = 1 if angle <= PI / 2 and angle >= -PI / 2 else -1
	var tg = tan(angle)
	var denominator = sqrt(pow(b, 2) + pow(a, 2) * pow(tan(angle), 2))
	var x = sign * a * b / denominator
	var y = sign * a * b * tg / denominator
	return Vector2(x, y)
