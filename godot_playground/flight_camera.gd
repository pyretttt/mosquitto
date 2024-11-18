extends MeshInstance3D

@export 
var current: bool:
	get:
		return Camera3D.current
	set(value):
		$Camera3D.current = true

# Called when the node enters the scene tree for the first time.
func _ready():
	pass


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
