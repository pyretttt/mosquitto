//
//  modify.swift
//  app
//
//  Created by Бакулин Семен Александрович on 24.02.2026.
//

import Foundation

public func modify<T>(_ object: T, mutation: (inout T) -> Void) -> T {
    var obj = object
    mutation(&obj)
    return obj
}
