# FIGHT ENGINE START
class_name FightEngine

extends RefCounted

var characters: Dictionary # Dictionary[Team, Array[Characters]]
var operations: Dictionary # Dictionary[Character, Array[AtomOp]]

#func _init():

func loop(dt: float):
	var is_char_alive = func (char: Character):
		char.current_hp > 0
	var red_team: Array[Character] = characters[Team.RED]
	var blue_team: Array[Character] = characters[Team.BLUE]
	var red_alive = red_team.filter(is_char_alive)
	var blue_alive = blue_team.filter(is_char_alive)
	
	for character: Character in operations: # It implies that all actions are simultaneous
		var ops: Array[AtomOp] = operations[character]
		var to_append: Array[AtomOp] = []
		var to_remove_idx: Array[int] = []
		for op_idx in range(ops.size()):
			var op = ops[op_idx]
			to_append.append_array(op.tick(dt, characters))
			if op.progress >= 1.0:
				to_remove_idx.append(op_idx)
			if op.is_exclusive:
				break
		
		to_remove_idx.reverse()
		for remove_idx in to_remove_idx:
			ops.remove_at(remove_idx)
		ops.append_array(to_append)

	for character: Character in red_alive:
		var actions = character.act(characters)
		operations[character] += actions
	
	for character: Character in blue_alive:
		var actions = character.act(characters)
		operations[character] += actions

## FIGHT ENGINE END

enum Team { RED, BLUE }

class Pair:
	var left: Variant
	var right: Variant

	func _init(left: Variant, right: Variant):
		self.left = left
		self.right = right

class AtomOp:
	var owner: Character
	var duration: float
	var progress: float
	var is_exclusive: bool
	var is_interruptable: bool
	
	var owner_id: int:
		get: return owner.id

	var is_owner_alive: bool:
		get: return owner.current_hp > 0
	
	func _init(
		owner: Character,
	 	duration: float,
		progress: float = 0.0,
		is_exclusive: bool = true,
		is_interruptable: bool = true
	):
		self.owner = owner
		self.duration = duration
		self.progress = progress
		self.is_exclusive = is_exclusive
		self.is_interruptable = is_interruptable

	func update_progress(dt: float): # in seconds
		var delta_progress = dt / duration
		progress = min(progress + delta_progress, 1.0)

	func tick(
		dt: float, # in seconds
		characters: Dictionary
	) -> Array[AtomOp]:
		update_progress(dt)
		return []

# Single iteration movement
class MoveOp extends AtomOp:
	func _init(
		owner: Character, 
		speed: Vector2, # per second
		duration: float = 1,
		is_one_shot: bool = true
	):
		super._init(owner, 1.0)
		self.speed = speed
		self.is_one_shot = is_one_shot
		
	func tick(
		dt: float, # in seconds
		characters: Dictionary
	) -> Array[AtomOp]:
		if not is_owner_alive:
			progress = INF
			return []
		if progress == 0.0:
			owner.animation_state = AnimationState.new(
				AnimationState.Type.MOVING,
				owner.move_speed,
				owner.config.run_anim_names,
				true
			)
		var ops = super.tick(dt, characters)
		owner.position += self.speed * dt
		if self.is_one_shot:
			progress = INF
		
		return ops

class AttackOp extends AtomOp:
	func _init(
		owner: Character, 
		target: Character,
		physical_damage: int,
		duration: float, # in sec,
		progress: float = 0.0,
		is_recurrent: bool = true
	):
		super._init(owner, duration, progress)
		self.target = target
		self.physical_damage = physical_damage
		self.is_recurrent = is_recurrent
	
	func tick(
		dt: float, # in seconds
		characters: Dictionary
	) -> Array[AtomOp]:
		if not is_owner_alive:
			progress = INF
			return []
		if progress == 0.0:
			owner.animation_state = CharacterOperationState.ATTACKING
		var ops = super.tick(dt, characters)
		if progress >= 1.0:
			self.target.get_damage(self.damage, 0)
		
		if self.is_recurrent: # Probably not needed
			var new_op: AtomOp = AttackOp.new(
				owner, 
				self.target, 
				self.physical_damage,
				duration,
				progress,
				true
			)
			self.progress = INF
			return ops + new_op
			
		return ops

class CharacterAbility:
	var cooldown: float # in seconds
	var cooldown_in_progress: float = 0
	
	func _init(
		cooldown: float,
		cooldown_in_progress: float = 0
	):
		self.cooldown = cooldown
		self.cooldown_in_progress = cooldown_in_progress
	
	func is_suitable(
		owner: Character,
		allies: Array[Character],
		enemies: Array[Character]
	) -> bool:
		if cooldown_in_progress < 1.0:
			return false
		return true
		
	func launch(
		owner: Character,
		allies: Array[Character],
		enemies: Array[Character]
	) -> Array[AtomOp]:
		cooldown_in_progress = 0.0
		return []


class CharacterConfig:
	var hp: int
	var move_speed: float
	var base_attack_damage: int
	var attack_range: float
	var run_anim: String
	var fight_anims: Array[String]
	var stance_anims: Array[String]
	var scene_name: String
	var abilities: Array[CharacterAbility]
	var attack_speed: float


class AnimationState:
	enum Type {
		IDLE,
		MOVING,
		ATTACKING,
		SPELLING,
		KNOCKED,
		DIYING
	}
	
	var type: Type
	var speed: float # from 0 to 2
	var animationName: String
	var is_looped: bool
	
	func _init(
		type: Type, 
		speed: float,
		animationName: String,
		is_looped: bool
	):
		self.type = type
		self.speed = speed
		self.animationName = animationName
		self.is_looped = is_looped
		
	
	
	

class Character:
	var id: int
	var team_id: Team
	var config: CharacterConfig
	
	# State
	var hps: Vector2i # Remaining/Total let's be int for a simplicity
	var physical_armor: int # 0 to 60
	var magic_resistance: int # 0 to 60
	var attack_damage: int
	var attack_speed: float
	var move_speed: float
	var crit_odds: float # TODO: Apply
	var abilities: Array[CharacterAbility]
	var position: Vector2
	
	signal animation_state_changed(old: AnimationState, new: AnimationState)
	var animation_state = FightEngine.AnimationState.new(
		AnimationState.Type.IDLE, 
		0.0,
		"",
		false
	):
		get:
			return animation_state
		set(value):
			var old_state = animation_state
			animation_state = value
			animation_state_changed.emit(old_state, value)

	signal health_changed(old: int, new: int)
	var current_hp:
		get: return hps.x
		set(value):
			var old_value = hps.x
			hps.x = max(0, hps.x) 
			health_changed.emit(old_value, hps.x)
	
	func _init(id: int, config: CharacterConfig, team_id: Team):
		pass
		

	func lvl_up():
		pass
	
	func get_damage(physical_damage: int, magic_damage: int):
		assert(current_hp > 0, "Character receive damage when not alive")
		if physical_damage > 0:
			var taken_physical_damage = roundi(physical_damage * (1 - physical_armor / 100))
			current_hp -= taken_physical_damage
		if magic_damage > 0 and current_hp > 0:
			var taken_magic_damage = roundi(magic_damage * (1 - magic_resistance / 100))
			current_hp -= taken_magic_damage

	func distance(to: Character) -> float:
		var direction: Vector2 = to.posisition - position
		return direction.length()


	func has_attack_target(enemies: Array[Character]) -> bool:
		for enemy in enemies:
			if config.attack_range >= distance(enemy):
				return true
		return false
		
	func attack_enemy(enemies: Array[Character]) -> Array[AtomOp]:
		var reachable_targets = enemies.filter(
			func (enemy: Character): 
				return config.attack_range >= distance(enemy)
		)
		assert(not reachable_targets.is_empty(), "Tried to attack when no enemies nearby")
		# Pick first for now
		var target = reachable_targets[0]
		
		return [
			FightEngine.AttackOp.new(
				weakref(self), 
				weakref(target),
			 	attack_damage,
				attack_speed
			)
		]
		
	func pick_an_enemy_to_move_to(enemies: Array[Character]):
		assert(not enemies.is_empty(), "Empty enemies set")
		var enemies_ = enemies.duplicate()
		enemies_.sort_custom(
			func (enemy1: Character, enemy2: Character):
				return distance(enemy1) < distance(enemy2)
		)

		# Pick first for now
		return enemies_[0]
		
	func move_to(character: Character) -> AtomOp:
		var direction = (character.posisition - position).normalized()
		return FightEngine.MoveOp.new(
			weakref(self), 
			direction * config.move_speed
		)
	
	func act(characters: Dictionary) -> Array[AtomOp]:
		var allies = characters[team_id]
		var enemies = characters[Team.BLUE if team_id == Team.RED else Team.RED]
		for ability in abilities:
			if ability.is_suitable(self, allies, enemies):
				return ability.launch(self, allies, enemies)
		
		if has_attack_target(enemies):
			return attack_enemy(enemies)
		else:
			var enemy = pick_an_enemy_to_move_to(enemies)
			return [move_to(enemy)]
