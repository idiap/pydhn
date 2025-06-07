# Change Log

## 0.1.3

### Added

* Added documentation files in /docs
* Added a GitHub action to deploy the documentation on Pages

### Changed

* Fixed a bug in pipe_test when reading local data
* docstring_parameters now ignores curly braces outside the Parameter section of a docstring
* Modified .pre-commit-config.yaml so that isort ignores init files to avoid circular import issues 
* Added dependencies needed for docs to pyproject.toml 

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
