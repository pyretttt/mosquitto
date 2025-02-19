class_name Configs

extends Object

static var hp_scale: int = 1500
static var move_speed_scale: float = 3.0
static var attack_speed_scale: float = 1.0
static var attack_damage_scale: int = 120
static var attack_range_scale: float = 4
static var armor_scale: int = 1.0
static var magic_resistance_scale: int = 1.0
static var crit_odds_scale: float = 1.0

const configs_path = "characters/configs"

class ConfigLoader:
	static func load_character_configs() -> Array[CharacterConfig]:
		const path := "res://" + configs_path
		if not DirAccess.dir_exists_absolute(path):
			assert(false, "Non existing configs directory")
		var file_names := DirAccess.get_files_at(path)
		var result: Array[CharacterConfig] = []
		for file in file_names:
			var config_path = path.path_join(file)
			var json = load(config_path)
			result.append(CharacterConfig.from_json(json.data))
			print("Loaded ", config_path, " config")
			
		return result

class FightConfig:
	enum Mode {
		DEMO,
		AUTOCHESS
	}
	var characters_configs: Array[CharacterConfig]
	var mode: Mode
	var max_characters_in_team: int
	
	func _init(
		characters_configs: Array[CharacterConfig],
		mode: Mode,
		max_characters_in_team: int
	):
		self.characters_configs = characters_configs
		self.mode = mode
		self.max_characters_in_team = max_characters_in_team


class CharacterConfig:
	enum CharacterClass { 
		WARRIOR, MAGE, RIFLEMAN, TANK
	}
	var name: String
	var scene_path: String
	var cls: CharacterClass
	
	var hp: float
	var move_speed: float
	var attack_damage: float
	var attack_range: float
	var attack_speed: float
	var armor: float
	var magic_resistance: float
	var crit_odds: float
	
	var run_anim: String
	var fight_anims: Array
	var stance_anims: Array
	var abilities: Array
	
	var level_up: LevelUpConfig
	
	static func from_json(character_config: Variant) -> CharacterConfig:
		var cfg = CharacterConfig.new()
		match character_config.class:
			"warrior": cfg.cls = CharacterClass.WARRIOR
			"mage": cfg.cls = CharacterClass.MAGE
			"rifleman": cfg.cls = CharacterClass.RIFLEMAN
			"tank": cfg.cls = CharacterClass.TANK
			var unknown: assert(false, "Unknown character class %s" % unknown)
		
		cfg.name = character_config.name
		cfg.scene_path = character_config.scene_path
		cfg.hp = character_config.hp * Configs.hp_scale
		cfg.move_speed = character_config.move_speed * Configs.move_speed_scale
		cfg.attack_damage = character_config.attack_damage * Configs.attack_damage_scale
		cfg.attack_range = character_config.attack_range * Configs.attack_range_scale
		cfg.attack_speed = character_config.attack_speed * Configs.attack_speed_scale
		cfg.armor = character_config.armor * Configs.armor_scale
		assert(cfg.armor <= 0.6, "Max armor is 0.6")
		cfg.magic_resistance = character_config.magic_resistance * Configs.magic_resistance_scale
		assert(cfg.magic_resistance <= 0.6, "Max magic resistance is 0.6")
		cfg.crit_odds = character_config.crit_odds * Configs.crit_odds_scale
		
		cfg.run_anim = character_config.run_anim
		print("typeof: ", type_string(typeof(character_config.fight_anims as Array[String])))
		cfg.fight_anims = Array(character_config.fight_anims)
		cfg.stance_anims = (character_config.stance_anims as Array[String])
		cfg.abilities = character_config.abilities.map(AbilityConfig.from_json)
		
		cfg.level_up = LevelUpConfig.from_json(character_config.level_up)
		return cfg

class LevelUpConfig:
	var hp_update: float
	var move_speed_update: float
	var attack_damage_update: float
	var attack_range_update: float
	var attack_speed_update: float
	var armor_update: int
	var magic_resistance_update: int
	var crit_odds_update: float
	
	static func from_json(cfg: Variant) -> LevelUpConfig:
		var obj = LevelUpConfig.new()
		obj.hp_update = cfg.hp_update * Configs.hp_scale
		obj.move_speed_update = cfg.move_speed_update * Configs.hp_scale
		obj.attack_damage_update = cfg.attack_damage_update * Configs.attack_damage_scale
		obj.attack_range_update = cfg.attack_range_update * Configs.attack_range_scale
		obj.attack_speed_update = cfg.attack_speed_update * Configs.attack_speed_scale
		obj.armor_update = cfg.armor_update * Configs.armor_scale
		obj.magic_resistance_update = cfg.magic_resistance_update * Configs.magic_resistance_scale
		obj.crit_odds_update = cfg.crit_odds_update * Configs.magic_resistance_scale

		return obj

class AbilityConfig:
	enum Type { 
		TARGET_PROJECTILE,
		AOE_PROJECTILE,
		BUFF
	}
	var type: Type
	var cooldown: float # in seconds
	var cast_duration: float # in seconds
	var is_spelling_interruptable: bool
	var wiped_on_death: bool
	const is_exclusive: bool = true # Just a marker to think about it later
	const scene_name: String = "" # Just a marker to think about it later
	
	static func from_json(cfg: Variant) -> AbilityConfig:
		# TODO: Implement later
		return AbilityConfig.new()
