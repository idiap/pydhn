## Contributing to PyDHN

Welcome\! This document outlines the process and best practices for contributing to the PyDHN project. Following these guidelines ensures a smooth and consistent development experience for everyone.

### 1\. Prerequisites

Before you start, make sure you have:

  * **Git** installed.
  * **Python** (`>=3.9,<3.13`) installed.
  * **Poetry** (our dependency manager and build tool) installed. If you don't have it, follow the official Poetry installation guide.

### 2\. Initial Setup

To set up your local development environment:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/idiap/pydhn.git
    cd pydhn
    ```

2.  **Install project dependencies with Poetry:**
    This command will create a virtual environment (if one doesn't exist) and install all required and development dependencies.

    ```bash
    poetry install
    ```

3.  **Install Git `pre-commit` hooks:**
    This command will install `pre-commit` to automate code quality checks (`black`, `isort`, `flake8`, etc.) before each commit.

    ```bash
    poetry run pre-commit install
    ```

### 3\. Development Workflow

1.  **Create a new branch:**
    Always create a new branch for your changes:

    ```bash
    git checkout -b feature/my-new-feature-name
    # Or: bugfix/fix-issue-description, chore/update-docs, etc.
    ```

2.  **Make your changes:**
    Write code, add tests, update documentation.

3.  **Run `pre-commit` hooks manually (optional):**
    To fix formatting and check for linting errors across all files (not just staged ones), run:

    ```bash
    poetry run pre-commit run --all-files
    ```

    This will automatically format your code. You'll then need to stage any files modified by these tools.

4.  **Run tests:**
    Ensure all tests pass before committing:

    ```bash
    python -m unittest discover
    ```

5.  **Commit your changes:**

      * Stage your changes: `git add .`
      * Commit using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): Your `pre-commit` hook should enforce this. Examples:
          * `feat: Add new feature`
          * `fix: Resolve bug in module X`
          * `chore: Update dependencies`
          * `docs: Improve README structure`

### 4\. Submitting Contributions

1.  **Push your branch:**

    ```bash
    git push origin your-branch-name
    ```

2.  **Open a Pull Request (PR):**
    On GitHub, navigate to your repository and open a new PR from your branch to `main`.

      * Provide a PR title.
      * Describe the changes you've made, why they were made, and any relevant issue numbers (e.g., `Closes #123`).

3.  **Address Feedback:**
    We'll work together during code review to refine your contribution. We encourage you to share your insights and iterate on changes to make your pull request even stronger.

### 5\. Troubleshooting

  * **`pre-commit` hook failures:** Read the error messages carefully. Fix the code, stage the changes (`git add .`), and `git commit` again.
