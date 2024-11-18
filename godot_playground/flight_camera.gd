extends MeshInstance3D

static var _current := true
@export 
var current: bool:
	get:
		return _current
	set(value):
		_current = value

# Called when the node enters the scene tree for the first time.
func _ready():
	$Camera3D.current = current


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
