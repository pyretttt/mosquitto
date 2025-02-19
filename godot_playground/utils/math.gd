extends Object
class_name Math

static func vec3_as_quat(vec: Vector3) -> Quaternion:
	return Quaternion(vec.x, vec.y, vec.z, 0)
	
static func vec3_from_quat(quat: Quaternion) -> Vector3:
	return Vector3(quat.x, quat.y, quat.z)

static func rotate_vec_with_quat(
	vec: Vector3,
	quat: Quaternion
) -> Vector3:
	assert(quat.is_normalized(), "Rotation quaternion must be unit length")
	return vec3_from_quat(quat * vec3_as_quat(vec) * quat.inverse())

static func apply_quat_to_basis(
	quat: Quaternion, 
	basis: Basis
) -> Basis:
	var new_basis = Basis.IDENTITY
	new_basis.x = rotate_vec_with_quat(basis.x, quat)
	new_basis.y = rotate_vec_with_quat(basis.y, quat)
	new_basis.z = rotate_vec_with_quat(basis.z, quat)
	
	return new_basis

static func quat_from(axis: Vector3, radian: float) -> Quaternion:
	var angle = radian / 2
	return Quaternion(
		sin(angle) * axis.x,
		sin(angle) * axis.y,
		sin(angle) * axis.z,
		cos(angle)
	)

static func quat_pure_from(vec: Vector3) -> Quaternion:
	return Quaternion(
		vec.x,
		vec.y,
		vec.z,
		0
	)
	
static func vec2_from_vec3(vec: Vector3) -> Vector2:
	return Vector2(vec.x, vec.y)
