# Updating all requirements

If you are creating a conda environment YAML file from an existing file, you may want to update all (or most) of the packages to the most recent versions. This is trickier than it might seem, since you ideally want to compile the requirements from inside the runtime container. The `Makefile` contains a few helpful commands to accomplish that. **Note: This typically only needs to be done once at the beginning of the competition―participants should focus on updating one or a few requirements at a time using the instructions [here](../README.md/#2-updating-the-runtime-packages).**

The first step is to "unpin" all of the package versions in the existing YAML files, i.e., go from something like `scipy=1.5.2=py38h8c5af15_0` to `scipy`. The following commands will overwrite the Python environment YAML files in `runtime`.

```bash
make unpin-python-requirements
```

Now you can edit the environment YAML files to pin any desired versions. Note that you can be as specific as you want, e.g., `scipy==1`, `scipy==1.5`, `scipy==1.5.2`, or `scipy=1.5.2=py38h8c5af15_0`.

The next commands will build the container, letting `conda` perform "dependency resolution", i.e., select specific versions of each package that all work together. The command also overwrite the existing environment YAML with a new file containing pinned versions.

```bash
make resolve-requirements
```

The new environment YAML file contains a _complete_ list of the packages in the environment (including subdependencies of the the specified packages). While this is great for reproducibility, it is a bit overdetermined―it increases the chance that any new package added will have a dependency conflict with the existing pinned packages. It is better to only pin the versions of the top-level packages you want, and then let conda dependency resolver find subdependencies that work with everything. Manually edit the environment YAML files to only include the pinned versions of the partial list of top-level packages you want to include.
