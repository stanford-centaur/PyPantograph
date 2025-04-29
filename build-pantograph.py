import os
import stat
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
import subprocess
import shutil
from pathlib import Path


class BuildRepl(BuildHookInterface):
    PLUGIN_NAME = "build_repl"
    PATH_PANTOGRAPH = Path("./src")
    PATH_PY = Path("./pantograph")

    def initialize(self, version, build_data):
        print("Building Pantograph REPL")
        print("Running: lake build repl")
        result = subprocess.run(
            args=["lake", "build", "repl"],
            cwd=self.PATH_PANTOGRAPH,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("Done")
        if result.returncode != 0:
            print(result.stderr)
            print(result.stdout)
            raise Exception(f"Failed to build Pantograph REPL: {result.stderr}")

        path_executable = self.PATH_PY / "pantograph-repl"

        # Copy the executable to the output directory
        repl_src = "repl.exe" if os.name == "nt" else "repl"
        shutil.copyfile(self.PATH_PANTOGRAPH / f".lake/build/bin/{repl_src}", path_executable)

        # Make the executable executable
        os.chmod(path_executable, os.stat(path_executable).st_mode | stat.S_IEXEC)

        # Copy the lean-toolchain file to the output directory
        shutil.copyfile(self.PATH_PANTOGRAPH / "lean-toolchain", self.PATH_PY / "lean-toolchain")

        return super().initialize(version, build_data)