extends RefCounted

var _elements: Array[Variant]

var is_empty: bool:
	get: return _elements.is_empty()

func _init(elements: Array[Variant] = []):
	self._elements = elements
	
func enque(element: Variant) -> void:
	self._elements.append(element)
	
func deque() -> Variant:
	if _elements.is_empty():
		return null
	
	return _elements.pop_front()
	
func clear() -> void:
	_elements = []
