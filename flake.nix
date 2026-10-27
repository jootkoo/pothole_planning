{
  description = "Pothole Detection and Autonomous Path Planning - Python Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems = function:
        nixpkgs.lib.genAttrs supportedSystems
          (system: function nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            python312
            git
          ];

          shellHook = ''
            echo "============================================"
            echo " Pothole Detection Development Environment"
            echo "============================================"
            echo "Python: $(python --version)"
            echo ""
            echo "To activate your virtual environment:"
            echo "  source .venv/bin/activate"
            echo ""
            echo "To install dependencies:"
            echo "  python -m pip install -r requirements.txt"
            echo ""
          '';
        };
      });
    };
}