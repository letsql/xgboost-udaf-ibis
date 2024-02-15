{
  description = "Application packaged using poetry2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        pkgs = nixpkgs.legacyPackages.${system};
        inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication;
        myapp = mkPoetryApplication {
          projectDir = self;
          preferWheels = true;
        };
      in
      {
        packages = {
          inherit myapp;
          default = self.packages.${system}.myapp;
        };

        devShells = {
          default = pkgs.mkShell {
            packages = [
              self.packages.${system}.myapp
              pkgs.poetry
            ];
          };
        };

        apps = {
          default = {
            type = "app";
            program = "${self.packages.${system}.myapp}/bin/xgboost-udaf-ibis";
          };
        };
      });
}
