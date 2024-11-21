extends Node3D

# Camera
var CAMERA_ANGLE_BEGIN: float
var CAMERA_BEGIN_POS: Vector3
const CAMERA_ELLIPSE_AB = Vector2(50, 27)
const CAMERA_LOOK_AT: Vector3 = Vector3.ZERO

const CAMERA_INTEROP_PROGRESS_PER_SE := PI / 4

# State
var cameraMovingDirection := 0
var camera_current_angle: float

# Called when the node enters the scene tree for the first time.
func _ready():
	setup_camera()
	camera_current_angle = CAMERA_ANGLE_BEGIN


func _process(delta):
	if (cameraMovingDirection != 0):
		var delta_angle = cameraMovingDirection * delta * CAMERA_INTEROP_PROGRESS_PER_SE
		camera_current_angle += delta_angle
		var camera_pos = camera_point(
			camera_current_angle, 
			CAMERA_ELLIPSE_AB.x, 
			CAMERA_ELLIPSE_AB.y
		)
		camera_pos = Vector3(camera_pos.x, CAMERA_BEGIN_POS.y, camera_pos.y)
		$FlightCamera.transform.origin = camera_pos
		var camera_look_at = CAMERA_LOOK_AT
		camera_look_at.x += camera_pos.x / 2
		$FlightCamera.look_at(camera_look_at)
	

func setup_camera():
	# TODO: to config
	CAMERA_ANGLE_BEGIN = atan(
		($FlightCamera.transform.origin.z / $FlightCamera.transform.origin.x)
		* (CAMERA_ELLIPSE_AB.x / CAMERA_ELLIPSE_AB.y)
	)
	CAMERA_BEGIN_POS = $FlightCamera.transform.origin
	camera_current_angle = CAMERA_ANGLE_BEGIN
	$FlightCamera.look_at(CAMERA_LOOK_AT)

	
func _input(event):
	if event is InputEventKey and (event.keycode == KEY_A or event.keycode == KEY_D):
		if event.is_pressed():
			cameraMovingDirection = 1 if event.keycode == KEY_D else -1
		else:
			cameraMovingDirection = 0
	elif event is InputEventMagnifyGesture:
		print(event)
		cameraMovingDirection = event.factor - 1


func camera_point(angle, a, b):
	var r_theta = a * b / sqrt(a * a * pow(sin(angle), 2) + b * b * pow(cos(angle), 2))
	return Vector2(
		r_theta * cos(angle),
		r_theta * sin(angle)
	)
