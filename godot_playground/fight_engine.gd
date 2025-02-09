# FIGHT ENGINE START
class_name FightEngine

extends RefCounted

var characters: Dictionary # Dictionary[Team, Array[Characters]]

func _init(
	config: Configs.FightConfig
):
	pass

func loop(dt: float):
	var is_char_alive = func (c: Character):
		return c.current_hp > 0
	var red_team: Array[Character] = characters[Team.RED]
	var blue_team: Array[Character] = characters[Team.BLUE]
	var red_alive = red_team.filter(is_char_alive)
	var blue_alive = blue_team.filter(is_char_alive)
	var all_alive = red_alive + blue_alive
	
	for character: Character in all_alive: # It implies that all actions are simultaneous
		var ops: Array[Operation] = character.ops
		var to_append: Array[Operation] = []
		var to_remove_idx: Array[int] = []
		for op_idx in range(ops.size()):
			var op = ops[op_idx]
			to_append.append_array(op.tick(dt, characters))
			if op.progress >= 1.0:
				to_remove_idx.append(op_idx)
			if op.is_exclusive:
				break

		for remove_idx in to_remove_idx:
			if ops[remove_idx].progress >= 1.0:
				ops[remove_idx].finalize(characters)
			ops[remove_idx].on_remove(characters, null)
			ops.remove_at(remove_idx)
		ops.append_array(to_append)

	for character: Character in all_alive:
		var actions = character.act(characters)
		character.ops.append_array(actions)

## FIGHT ENGINE END

enum Team { RED, BLUE }

class Operation:
	var owner: WeakRef # Convert to weakref
	var duration: float
	var progress: float
	var is_exclusive: bool
	var is_interruptable: bool
	
	var owner_id: int:
		get: return strong_owner.id

	var is_owner_alive: bool:
		get: return strong_owner.current_hp > 0
		
	var strong_owner: Character:
		get: return owner.get_ref()
	
	func _init(
		owner: WeakRef,
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
		assert(duration > 0.0, "Zero or less duration is error")
		var delta_progress = dt / duration
		progress = min(progress + delta_progress, 1.0)
	
	func finalize(characters: Dictionary): # Maybe add -> Array[Operation]
		pass
		
	func on_remove(characters: Dictionary, cause: Operation): # Maybe add -> Array[Operation] and change cause: Operation -> Variant[String, Operation]
		pass
	
	func tick(
		dt: float, # in seconds
		characters: Dictionary
	) -> Array[Operation]:
		update_progress(dt)
		return []

class MoveOp extends Operation:
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
		operation_map: Dictionary
	) -> Array[Operation]:
		if progress == 0.0:
			strong_owner.animation_state = AnimationState.new(
				AnimationState.Type.MOVING,
				strong_owner.move_speed,
				strong_owner.config.run_anim_names,
				true
			)
		var ops = super.tick(dt, operation_map)
		strong_owner.position += self.speed * dt
		if self.is_one_shot:
			progress = INF
		
		return ops

class AttackOp extends Operation:
	var target: WeakRef
	
	var strong_target: Character:
		get: return target.get_ref() as Character
		
	var failed: bool = false
	
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
	
	func finalize(characters: Dictionary):
		if not failed:
			# TODO: Add critical damage test
			strong_target.get_damage(self.physical_damage, 0, self)
			
	func update_progress(dt: float):
		if not strong_target.current_hp > 0:
			progress = INF
		elif strong_owner.distance(strong_target) > strong_owner.config.attack_range * 2:
			progress = INF
		else:
			super.update_progress(dt)
	
	func tick(
		dt: float, # in seconds
		operation_map: Dictionary
	) -> Array[Operation]:
		if progress == 0.0:
			strong_owner.animation_state = AnimationState.new(
				AnimationState.Type.ATTACKING,
				strong_owner.move_speed,
				strong_owner.config.fight_anims.pick_random(),
				true
			)
		return super.tick(dt, operation_map)


class DeathOp extends Operation:
	func _init(
		owner: WeakRef
	):
		super._init(owner, 1.0, 1.0, true, false)
	
	func tick(dt: float, characters: Dictionary):
		# TODO: Implement
		return []


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
	) -> Array[Operation]:
		cooldown_in_progress = 0.0
		return []

class AnimationState:
	enum Type {
		IDLE,
		MOVING,
		ATTACKING,
		SPELLING,
		STUNNED,
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
	var config: Configs.CharacterConfig
	
	# State
	var hps: Vector2i # Remaining/Total let's be int for a simplicity
	var armor: float # 0 to 0.6
	var magic_resistance: float # 0 to 0.6
	var attack_damage: int
	var attack_speed: float
	var move_speed: float
	var crit_odds: float # TODO: Apply
	var abilities: Array[CharacterAbility]
	var position: Vector2
	var ops: Array[Operation]
	
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
	var current_hp: int:
		get: return hps.x
		set(value):
			var old_value = hps.x
			hps.x = max(0, hps.x) 
			health_changed.emit(old_value, hps.x)
	
	func _init(
		id: int, 
		config: Configs.CharacterConfig, 
		team_id: Team, 
		position: Vector2 = Vector2.ZERO
	):
		self.id = id
		self.team_id = team_id
		self.config = config
		self.hps = Vector2i(config.hp, config.hp)
		self.physical_armor = config.physical_armor
		self.magic_resistance = config.magic_resistance
		self.attack_damage = config.attack_damage
		self.attack_speed = attack_speed
		self.move_speed = move_speed
		self.crit_odds = crit_odds
		self.abilities = [] # TODO: Implement later
		self.position = position
		self.ops = ops # Enrich with passive abilities

	func lvl_up():
		self.hps.y += config.level_up.hp_update
		self.armor = min(config.level_up.armor_update + armor, 60)
		self.magic_resistance += config.level_up.magic_resistance_update
		self.attack_damage += config.level_up.attack_damage_update
		self.attack_speed += config.level_up.attack_speed_update
		self.move_speed += config.level_up.move_speed_updates
		self.crit_odds += config.level_up.crit_odds_updates
	
	func get_damage(physical_damage: int, magic_damage: int, cause: Operation):
		assert(current_hp > 0, "Character receive damage when not alive")
		if physical_damage > 0:
			var taken_physical_damage = roundi(physical_damage * armor)
			current_hp -= taken_physical_damage
		if magic_damage > 0 and current_hp > 0:
			var taken_magic_damage = roundi(magic_damage * magic_resistance)
			current_hp -= taken_magic_damage

	func distance(to: Character) -> float:
		var direction: Vector2 = to.posisition - position
		return direction.length()

	func has_attack_target(enemies: Array[Character]) -> bool:
		for enemy in enemies:
			if config.attack_range >= distance(enemy):
				return true
		return false
		
	func attack_enemy(enemies: Array[Character]) -> Array[Operation]:
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
				target,
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
		
	func move_to(character: Character) -> Operation:
		var direction = (character.posisition - position).normalized()
		return FightEngine.MoveOp.new(
			weakref(self), 
			direction * config.move_speed
		)
	
	func cancel_all_operations(characters: Dictionary, cause: Operation):
		for op: Operation in ops: # Maybe reverse order
			op.on_remove(characters, cause)
		ops.clear()
	
	func act(characters: Dictionary) -> Array[Operation]:
		if current_hp == 0: # act in such state should be called once
			var death_op = DeathOp.new(weakref(self))
			cancel_all_operations(characters, death_op)
			return [death_op]
			
		if not ops.is_empty() and ops[0].is_exclusive:
			return [] # probably convert to null
		
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
