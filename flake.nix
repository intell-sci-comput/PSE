{
  description = "PSE - Symbolic Regression with MCTS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [(import rust-overlay)];

        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };
        rust = pkgs.rust-bin.selectLatestNightlyWith (
          toolchain:
            toolchain.default.override {
              extensions = [
                "rust-src"
                "rust-analyzer"
                "miri"
                "llvm-tools-preview"
              ];
              targets = ["x86_64-unknown-linux-gnu"];
            }
        );

        python = pkgs.python312;

        # Python dependencies from requirements.txt
        pythonEnv = python.withPackages (ps: with ps; [
          click
          deap
          matplotlib
          numba
          numpy
          pandas
          pyyaml
          scikit-learn
          scipy
          seaborn
          sympy
          tqdm
          symengine
          pip
          torch
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rust
            pythonEnv
            symengine
            gcc
            cmake
          ];
        };

        packages.default = pythonEnv;
      }
    );
}
