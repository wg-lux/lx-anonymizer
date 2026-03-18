import importlib


def test_core_modules_import() -> None:
    module_names = [
        "lx_anonymizer",
        "lx_anonymizer.cli",
        "lx_anonymizer.config",
        "lx_anonymizer.settings",
        "lx_anonymizer._native",
        "lx_anonymizer.sensitive_meta_interface",
        "lx_anonymizer.region_processing.box_operations",
        "lx_anonymizer.setup.custom_logger",
    ]

    for module_name in module_names:
        assert importlib.import_module(module_name) is not None
