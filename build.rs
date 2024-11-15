use std::{env, path::PathBuf, process::Command};

fn main() {
    // get git version info
    let output = Command::new("git")
        .args(&["describe", "--tags", "--always", "--dirty"])
        .output()
        .unwrap();
    let git_hash = String::from_utf8(output.stdout).unwrap();
    println!("cargo:rustc-env=GIT_HASH={}", git_hash);
    
    // ensure ffmpeg is available
    let ffmpeg_check = Command::new("ffmpeg")
        .arg("-version")
        .output();
    
    if ffmpeg_check.is_err() {
        println!("cargo:warning=ffmpeg not found in PATH, some features may not work");
    }

    // get output directory
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // compile ASR service proto files
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        // add properties to generated types
        .type_attribute("asr.TranscribeRequest", "#[derive(Hash)]")
        .type_attribute("asr.TranscribeResponse", "#[derive(Hash)]")
        // generate file descriptor
        .file_descriptor_set_path(out_dir.join("asr_descriptor.bin"))
        // compile proto files
        .compile_protos(&["proto/asr.proto"], &["proto"])
        .unwrap_or_else(|e| panic!("Failed to compile protos: {}", e));

    // if need to generate to specific directory (optional)
    let pb_dir = PathBuf::from("src/grpc/pb");
    if !pb_dir.exists() {
        std::fs::create_dir_all(&pb_dir).unwrap();
    }
    
    // configure compile to specific directory
    tonic_build::configure()
        .out_dir(pb_dir)
        // add properties to server code
        .server_mod_attribute("asr", "#[cfg(feature = \"server\")]")
        // add properties to client code
        .client_mod_attribute("asr", "#[cfg(feature = \"client\")]")
        .compile_protos(&["proto/asr.proto"], &["proto"])
        .unwrap_or_else(|e| panic!("Failed to compile protos to specific dir: {}", e));

    // notify Cargo to rerun if source files change
    println!("cargo:rerun-if-changed=proto/asr.proto");
    println!("cargo:rerun-if-changed=build.rs");
}