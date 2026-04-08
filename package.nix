{
  pkgs,
  rustPlatform,
  pkg-config,
  lib,
  python312Packages,
  fetchPypi,
  ffmpeg-headless,
  tesseract,
  ollama,
  withLlm ? false,
  withOcr ? false,
  withNlu ? false,
}:

let
  py = python312Packages;
  pname = "lx-anonymizer";
  version = "0.9.0.6";
  tesseractWithLangs = tesseract.override {
    enableLanguages = [
      "deu"
      "eng"
    ];
  };

  gender-guesser = py.buildPythonPackage rec {
    pname = "gender-guesser";
    version = "0.4.0";
    pyproject = false;

    src = fetchPypi {
      inherit pname version;
      hash = "sha256-FZHBRZKAXKfaBqRtX3ICUR98uHVHBJpo38y+7bh58xs=";
    };
  };

  src = lib.cleanSourceWith {
    src = ./.;
    filter =
      path: type:
      let
        rel = lib.removePrefix "${toString ./.}/" (toString path);
        base = builtins.baseNameOf (toString path);
        ignoredBaseNames = [
          ".devenv"
          ".direnv"
          ".env"
          ".git"
          ".mypy_cache"
          ".pytest_cache"
          ".ruff_cache"
          ".venv"
          "__pycache__"
        ];
        ignoredPrefixes = [
          "logs/"
        ];
      in
      !(builtins.elem base ignoredBaseNames || lib.any (prefix: lib.hasPrefix prefix rel) ignoredPrefixes);
  };

  baseDeps = with py; [
    certifi
    dateparser
    pydantic
    pydantic-settings
    rapidfuzz
    requests
    tenacity
    types-requests
  ] ++ [
    gender-guesser
  ];

  imagePdfDeps = with py; [
    faker
    numpy
    opencv-python-headless
    pdfplumber
    pillow
    protobuf
    pymupdf
  ];

  llmDeps = with py; [
    py.ollama
    tiktoken
  ];

  ocrDeps = with py; [
    pytesseract
    tesserocr
    torch
    torchaudio
    torchvision
    transformers
  ];

  nluDeps = with py; [
    pyspellchecker
    spacy
  ];

  selectedDeps =
    baseDeps
    ++ lib.optionals (withOcr || withNlu) imagePdfDeps
    ++ lib.optionals withLlm llmDeps
    ++ lib.optionals withOcr ocrDeps
    ++ lib.optionals withNlu nluDeps;
in
py.buildPythonPackage {
  inherit pname version src;
  pyproject = true;
  dontCheckRuntimeDeps = true;
  cargoDeps = rustPlatform.fetchCargoVendor {
    inherit pname version src;
    hash = "sha256-3srOZAciN512kYJ9eQ1vzAQQu1GibxdpL3dlV6c5w3w=";
  };

  nativeBuildInputs = [
    rustPlatform.maturinBuildHook
    rustPlatform.cargoSetupHook
  ] ++ [
    pkg-config
  ];

  propagatedBuildInputs = selectedDeps;

  buildInputs = [
    ffmpeg-headless
  ]
  ++ lib.optionals withOcr [ tesseractWithLangs ]
  ++ lib.optionals withLlm [ ollama ];

  pythonImportsCheck = [
    "lx_anonymizer"
    "lx_anonymizer.cli"
    "lx_anonymizer.settings"
  ];

  installPhase = ''
    runHook preInstall

    mkdir -p "$out/$pythonSitePackages" "$out/bin"
    python - <<'PY'
import pathlib
import zipfile

wheel_path = next(pathlib.Path("dist").glob("*.whl"))
with zipfile.ZipFile(wheel_path) as zf:
    zf.extractall("wheel-unpack")
PY
    cp -r wheel-unpack/lx_anonymizer wheel-unpack/*.dist-info "$out/$pythonSitePackages/"

    cat > "$out/bin/lx-anonymizer" <<EOF
#!${pkgs.runtimeShell}
export PATH="${lib.makeBinPath ([ ffmpeg-headless ] ++ lib.optionals withLlm [ ollama ])}:\$PATH"
export PYTHONPATH="$out/$pythonSitePackages''${PYTHONPATH:+:$PYTHONPATH}"
export TESSDATA_PREFIX="${tesseractWithLangs}/share/tessdata"
export OLLAMA_HOST="''${OLLAMA_HOST:-127.0.0.1:11434}"
exec ${py.python.interpreter} -m lx_anonymizer.cli "\$@"
EOF
    chmod +x "$out/bin/lx-anonymizer"

    runHook postInstall
  '';

  meta = with lib; {
    description = "OCR-driven anonymization pipeline for medical reports and endoscopy frames";
    homepage = "https://github.com/wg-lux/lx-anonymizer";
    license = licenses.mit;
    platforms = platforms.linux;
    mainProgram = "lx-anonymizer";
  };
}
