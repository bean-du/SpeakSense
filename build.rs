use std::{env, path::PathBuf, process::Command};

fn main() {
    // 获取 git 版本信息
    let output = Command::new("git")
        .args(&["describe", "--tags", "--always", "--dirty"])
        .output()
        .unwrap();
    let git_hash = String::from_utf8(output.stdout).unwrap();
    println!("cargo:rustc-env=GIT_HASH={}", git_hash);
    
    // 确保 ffmpeg 可用
    let ffmpeg_check = Command::new("ffmpeg")
        .arg("-version")
        .output();
    
    if ffmpeg_check.is_err() {
        println!("cargo:warning=ffmpeg not found in PATH, some features may not work");
    }

    // 获取输出目录
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // 编译 ASR 服务的 proto 文件
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        // 可以为生成的类型添加属性
        .type_attribute("asr.TranscribeRequest", "#[derive(Hash)]")
        .type_attribute("asr.TranscribeResponse", "#[derive(Hash)]")
        // 生成文件描述符
        .file_descriptor_set_path(out_dir.join("asr_descriptor.bin"))
        // 编译 proto 文件
        .compile_protos(&["proto/asr.proto"], &["proto"])
        .unwrap_or_else(|e| panic!("Failed to compile protos: {}", e));

    // 如果需要生成到特定目录（可选）
    let pb_dir = PathBuf::from("src/grpc/pb");
    if !pb_dir.exists() {
        std::fs::create_dir_all(&pb_dir).unwrap();
    }
    
    // 为特定目录的编译配置
    tonic_build::configure()
        .out_dir(pb_dir)
        // 可以为服务器端代码添加特性属性
        .server_mod_attribute("asr", "#[cfg(feature = \"server\")]")
        // 可以为客户端代码添加特性属性
        .client_mod_attribute("asr", "#[cfg(feature = \"client\")]")
        .compile_protos(&["proto/asr.proto"], &["proto"])
        .unwrap_or_else(|e| panic!("Failed to compile protos to specific dir: {}", e));

    // 通知 Cargo 在源文件改变时重新运行
    println!("cargo:rerun-if-changed=proto/asr.proto");
    println!("cargo:rerun-if-changed=build.rs");
}