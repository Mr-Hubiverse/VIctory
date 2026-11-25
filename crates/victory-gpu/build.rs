fn main() {
    #[cfg(windows)]
    {
        println!("cargo:rerun-if-changed=src/pipeline/seh_guard.c");
        cc::Build::new()
            .file("src/pipeline/seh_guard.c")
            .compile("victory_seh_guard");
    }
}
