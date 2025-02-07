# FIGHT ENGINE START
class_name FightEngine

extends RefCounted

var characters: Dictionary
var operations: Dictionary # Dictionary[Character, Array[AtomOp]]

#func _init():
	

## FIGHT ENGINE END

enum Team { RED, BLUE }

class Pair:
	var left: Variant
	var right: Variant

	func _init(left: Variant, right: Variant):
		self.left = left
		self.right = right

class AtomOp:
	var owner: WeakRef
	var duration: float
	var progress: float
	var is_exclusive: bool
	var is_owner_alive: bool:
		get: return owner.get_ref() != null
	
	func _init(
		owner: WeakRef,
	 	duration: float,
		progress: float = 0.0,
		is_exclusive: bool = true
	):
		self.owner = owner
		self.duration = duration
		self.progress = progress
		self.is_exclusive = is_exclusive

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
		owner: WeakRef, 
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
			owner.get_ref().op_state = CharacterOperationState.ATTACKING
		var ops = super.tick(dt, characters)
		owner.get_ref().position += self.speed * dt
		if self.is_one_shot:
			progress = INF
		
		return ops

class AttackOp extends AtomOp:
	func _init(
		owner: WeakRef, 
		target: WeakRef,
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
			owner.get_ref().op_state = CharacterOperationState.ATTACKING
		var ops = super.tick(dt, characters)
		if progress >= 1.0:
			var target_: Character = self.target.get_ref() as Character
			if target_ != null:
				target_.get_damage(self.damage, 0)
		
		if self.is_recurrent:
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
	var run_anim_names: String
	var fight_anim: Array[String]
	var stance_anim: Array[String]
	var scene_name: String
	var abilities: Array[CharacterAbility]
	var attack_speed: float


enum CharacterOperationState {
	IDLE,
	MOVING,
	ATTACKING,
	SPELLING,
	KNOCKED
}

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
	var crit_odds: float
	var abilities: Array[CharacterAbility]
	var position: Vector2
	
	signal op_state_changed(old: CharacterOperationState, new: CharacterOperationState)
	var op_state = CharacterOperationState.IDLE:
		get:
			return op_state
		set(value):
			var old_state = op_state
			op_state = value
			op_state_changed.emit(old_state, value)

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
