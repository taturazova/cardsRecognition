import subprocess
import pkg_resources

installed_packages = {pkg.key for pkg in pkg_resources.working_set}
for package in installed_packages:
    subprocess.call(["pip", "uninstall", "-y", package])
