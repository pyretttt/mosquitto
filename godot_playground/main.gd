extends Node3D

### To fight engine scene

var configs: Configs.FightConfig = Configs.FightConfig.new(
	Configs.ConfigLoader.load_character_configs(),
	Configs.FightConfig.Mode.DEMO,
	3
)

var fight_engine := FightEngine.new(
	configs,
	Vector3(-20, 1, 0),
	Vector3(20, 1, 0),
)
var character_nodes: Dictionary # [int, Node3d]
@onready 
var character_scenes: Dictionary = create_scenes(configs.characters_configs) # [String, Scene] name to scene

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
	
	character_nodes = init_characters_scenes()
	initial_layout_characters()
	
	#$Kaila.position = Vector3.ZERO
	#var kaila_run : Animation= $Kaila/kaila_r2u/AnimationPlayer.get_animation("stance")
	#kaila_run.loop_mode = Animation.LOOP_LINEAR
	#$Kaila/kaila_r2u/AnimationPlayer.play("stance")


func _process(delta):
	fight_engine.loop(delta)
	update_characters()
	
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
		
		# Light ray rotation
		$DirectionalLight3D.rotate(Vector3.UP, delta_angle)
		#$Kaila/kaila_r2u/AnimationPlayer.play("run")


func _input(event):
	if event is InputEventKey and (event.keycode == KEY_A or event.keycode == KEY_D):
		if event.is_pressed():
			cameraMovingDirection = 1 if event.keycode == KEY_A else -1
		else:
			cameraMovingDirection = 0
	elif event is InputEventMagnifyGesture:
		print(event)
		cameraMovingDirection = event.factor - 1


func setup_camera():
	CAMERA_ANGLE_BEGIN = atan(
		($FlightCamera.transform.origin.z / $FlightCamera.transform.origin.x)
		* (CAMERA_ELLIPSE_AB.x / CAMERA_ELLIPSE_AB.y)
	)
	CAMERA_BEGIN_POS = $FlightCamera.transform.origin
	camera_current_angle = CAMERA_ANGLE_BEGIN
	$FlightCamera.look_at(CAMERA_LOOK_AT)


func camera_point(angle, a, b):
	var r_theta = a * b / sqrt(a * a * pow(sin(angle), 2) + b * b * pow(cos(angle), 2))
	return Vector2(
		r_theta * cos(angle),
		r_theta * sin(angle)
	)
	
	
## To separate fight engine scene
func create_scenes(configs: Array[Configs.CharacterConfig]) -> Dictionary: # [Team, Character]
	var result = {}
	for config: Configs.CharacterConfig in configs:
		var scene_path = "res://".path_join(config.scene_path)
		var character_scene = load(scene_path)
		result[config.name] = character_scene
		
	return result
	
func init_characters_scenes() -> Dictionary:
	var characters: Dictionary = fight_engine.get_characters()
	var read_team: Array = characters[FightEngine.Team.RED]
	var blue_team: Array = characters[FightEngine.Team.BLUE]
	var result = {}
	for character: FightEngine.Character in read_team + blue_team:
		var scene: Resource = character_scenes[character.config.name]
		if scene.can_instantiate():
			result[character.id] = scene.instantiate()
		else:
			assert(
				false, 
				"Unable to create scene for character: %s".format(character.config.name)
			)
	
	return result

func initial_layout_characters():
	var characters: Dictionary = fight_engine.get_characters()
	var read_team: Array = characters[FightEngine.Team.RED]
	var blue_team: Array = characters[FightEngine.Team.BLUE]
	
	for character: FightEngine.Character in read_team + blue_team:
		var char_node: Node3D = character_nodes[character.id]
		add_child(char_node)
		char_node.position = character.position
		
		# TODO: Remove after map scale adjusting
		char_node.scale = Vector3(3, 3, 3)

func update_characters():
	var characters: Dictionary = fight_engine.get_characters()
	var read_team: Array = characters[FightEngine.Team.RED]
	var blue_team: Array = characters[FightEngine.Team.BLUE]
	
	for character: FightEngine.Character in read_team + blue_team:
		var char_node: Node3D = character_nodes[character.id]
		char_node.position = character.position
		char_node.get_node("kaila_r2u/AnimationPlayer").play("run")
