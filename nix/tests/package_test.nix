{
  pkgs,
  lxAnonymizer,
  ...
}: {
  suites."package".tests = [
    {
      name = "package-smoke";
      type = "script";
      script = ''
        export PATH="${pkgs.lib.makeBinPath [ pkgs.coreutils pkgs.gnugrep ]}"

        test -x ${lxAnonymizer}/bin/lx-anonymizer
        test -f ${lxAnonymizer}/${pkgs.python312.sitePackages}/lx_anonymizer/__init__.py

        ${lxAnonymizer}/bin/lx-anonymizer --help | grep -F "Run the LX Anonymizer image/PDF processing pipeline."
      '';
    }
    {
      name = "base-cli-missing-ocr-deps";
      type = "script";
      script = ''
        export PATH="${pkgs.lib.makeBinPath [ pkgs.coreutils pkgs.gnugrep ]}"

        set +e
        output="$(${lxAnonymizer}/bin/lx-anonymizer -i input.png 2>&1)"
        status=$?
        set -e

        test "$status" -eq 2
        printf '%s\n' "$output" | grep -F "Missing optional dependency for CLI pipeline"
        printf '%s\n' "$output" | grep -F "Install with \`pip install lx-anonymizer[ocr]\`."
      '';
    }
  ];
}
