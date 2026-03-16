import Foundation

public struct Pair<L, R> {
    public var left: L
    public  var right: R
    
    public init(left: L, right: R) {
        self.left = left
        self.right = right
    }
}

extension Pair: Equatable where L: Equatable, R: Equatable {}
extension Pair: Hashable where L: Hashable, R: Hashable {}
extension Pair: Sendable where L: Sendable, R: Sendable {}

public typealias IdentifiableValue<ID, Value> = Pair<ID, Value>

public func makeIdentifiable<V>(value: V, id: UUID = UUID()) -> IdentifiableValue<UUID, V> {
    Pair(left: id, right: value)
}
