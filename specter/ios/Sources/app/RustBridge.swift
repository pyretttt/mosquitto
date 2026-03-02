import Foundation
import rust_ffi

enum RustBridge {
    static func hello() -> String {
        String(cString: rust_hello())
    }
    
    static func number() -> Int32 {
        get_number()
    }
}
