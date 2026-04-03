uniffi::setup_scaffolding!();

#[uniffi::export]
fn hello_uniffi() -> String {
    "Hello from UniFFI Rust!".to_string()
}

#[uniffi::export]
fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}
