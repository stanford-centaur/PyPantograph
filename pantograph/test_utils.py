import subprocess
import sys
import textwrap


def test_utils_import_is_clean_with_deprecation_warnings_as_errors():
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-c",
            "import pantograph.utils",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_utils_get_event_loop_preserves_current_event_loop():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import asyncio

                custom_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(custom_loop)

                import pantograph.utils as utils

                try:
                    assert utils.get_event_loop() is custom_loop
                finally:
                    custom_loop.close()
                """
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
