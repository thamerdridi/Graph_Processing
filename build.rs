use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/kernels/bfs.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/bellman_ford.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/pagerank.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/label_propagation.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let cuda_files = [
        "src/cuda/kernels/bfs.cu",
        "src/cuda/kernels/bellman_ford.cu",
        "src/cuda/kernels/pagerank.cu",
        "src/cuda/kernels/label_propagation.cu",
    ];

    let mut object_files = Vec::new();

    for cuda_file in &cuda_files {
        let output_name = PathBuf::from(cuda_file)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let output_obj = out_dir.join(format!("{}.o", output_name));

        let status = Command::new("nvcc")
            .args(&[
                "-c",
                cuda_file,
                "-o",
                output_obj.to_str().unwrap(),
                "--compiler-options",
                "-fPIC",
                "-O3",
                "-arch=sm_86",
            ])
            .status()
            .expect("nvcc not found. Install CUDA toolkit.");

        if !status.success() {
            panic!("nvcc failed to compile {}", cuda_file);
        }

        object_files.push(output_obj);
    }

    let lib_path = out_dir.join("libcuda_kernels.a");
    Command::new("ar")
        .args(&["crus", lib_path.to_str().unwrap()])
        .args(&object_files)
        .status()
        .expect("Failed to create static library");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=cuda_kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
