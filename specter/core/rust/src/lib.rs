use std::os::raw::c_char;

// Static, null-terminated string that can be shared with C/Swift.
static RUST_HELLO: &[u8] = b"Hello from Rust!\0";

#[no_mangle]
pub extern "C" fn rust_hello() -> *const c_char {
    RUST_HELLO.as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn get_number() -> i32 {
    223
}
