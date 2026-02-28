{
  description = "PSE - Symbolic Regression with MCTS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

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
        ]);

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.symengine
            pkgs.gcc
            pkgs.cmake
          ];

          # shellHook = ''
          #   echo "PSE development environment"
          #   echo "Python: $(python --version)"
          #   echo ""
          #   echo "Note: Some packages may need to be installed via pip:"
          #   echo "  pip install dysts pysindy derivative"
          # '';
        };

        packages.default = pythonEnv;
      }
    );
}
