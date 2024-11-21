extends MeshInstance3D

@onready 
var camera := $Camera3D

var current: bool:
	get:
		return camera.current
	set(value):
		camera.current = true

# Called when the node enters the scene tree for the first time.
func _ready():
	pass


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
