# Change Log

## Unreleased

### Added

* Added a warning when a thermal simulation runs without a specified `ts_id`.
* Added a new test (`tests/test_dynamic_thermal_balance.py`) to verify thermal energy balance in dynamic simulations.
* Added a new example (`examples/dynamic_simulation.py`) for dynamic simulations with `LagrangainPipe`.

### Changed

* Fixed a bug in `pipe_test` related to local data reading.
* **Heat exchanger model:** Refined the heat exchanger model's behavior (`compute_hx_temp` and related functions) for consistent energy interpretation in dynamic simulations. The `delta_q` parameter and return values are now uniformly treated as Watt-hours (Wh). A new `stepsize` parameter (in seconds), defaulting to `3600.0`s for backward compatibility, was added for internal energy-to-power conversions. Users with non-hourly `stepsize` in dynamic simulations should now explicitly pass their `stepsize` to consumers and producers. (See #5)
* Changed the sign of `LagrangianPipe`'s `delta_q` to be negative when heat is lost.
* Updated documentation reflecting changes from #5.
* Updated default `stepsize` to `3600.0`.

### Removed


## 0.1.3

Released on October 30, 2024

### Added

* Added documentation files in /docs
* Added a GitHub action to deploy the documentation on Pages

### Changed

* Improved docstrings and type hints
* docstring_parameters now ignores curly braces outside the Parameter section of a docstring
* Modified .pre-commit-config.yaml so that isort ignores init files to avoid circular import issues 
* Added dependencies needed for docs to pyproject.toml 

### Removed


## 0.1.2

Released on March 19, 2024

### Added

### Changed

* Fixed a bug where the wrong vector function was called when using LagrangianPipe

### Removed


## 0.1.1

Released on January 19, 2024

### Added

* Added dynamic pipe from [Boghetti et al.](https://doi.org/10.1016/j.energy.2023.130169) with:
	- Verification on experimental data from [Schweiger et al.](https://doi.org/10.1016/j.energy.2018.08.193)
	- Tests
* Added new default values
* Added CHANGELOG.md
* Added decorator for dynamic default values in docstring

### Changed

* Updated Pipe's docstring

### Removed


## 0.1.0

Released on January 12, 2024

* Initial release
