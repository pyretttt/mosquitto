#!/usr/bin/env bash
# Fake cargo binary for uniffi-bindgen's `cargo metadata` call.
# Returns minimal JSON so uniffi-bindgen can locate the crate directory.
# Requires FAKE_CARGO_MANIFEST_PATH and FAKE_CARGO_CRATE_NAME env vars.
set -euo pipefail

if [[ "${1:-}" == "metadata" ]]; then
    MANIFEST_DIR="$(dirname "$FAKE_CARGO_MANIFEST_PATH")"
    cat <<JSON
{"packages":[{"name":"${FAKE_CARGO_CRATE_NAME}","version":"0.1.0","id":"path+file://${MANIFEST_DIR}#${FAKE_CARGO_CRATE_NAME}@0.1.0","source":null,"manifest_path":"${FAKE_CARGO_MANIFEST_PATH}","dependencies":[],"targets":[{"kind":["lib"],"crate_types":["lib"],"name":"${FAKE_CARGO_CRATE_NAME}","src_path":"${MANIFEST_DIR}/src/lib.rs","edition":"2021","doc":true,"doctest":true,"test":true}],"features":{},"edition":"2021","links":null,"license":null,"license_file":null,"metadata":null,"publish":null,"authors":[],"categories":[],"keywords":[],"readme":null,"repository":null,"homepage":null,"documentation":null,"rust_version":null,"default_run":null}],"workspace_members":["path+file://${MANIFEST_DIR}#${FAKE_CARGO_CRATE_NAME}@0.1.0"],"workspace_default_members":["path+file://${MANIFEST_DIR}#${FAKE_CARGO_CRATE_NAME}@0.1.0"],"resolve":null,"target_directory":"${MANIFEST_DIR}/target","version":1,"workspace_root":"${MANIFEST_DIR}","metadata":null}
JSON
    exit 0
fi

echo "fake_cargo: unsupported subcommand: $*" >&2
exit 1
