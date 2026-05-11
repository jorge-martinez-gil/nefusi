# Contributing to NEFUSI

Thank you for your interest in contributing to NEFUSI! Below you will find guidance on how to report issues, suggest improvements, and submit code changes.

## Reporting Bugs

If you find a bug, please [open a GitHub Issue](https://github.com/jorge-martinez-gil/nefusi/issues/new) with the label **`bug`**. Include:

- A clear description of the problem
- Steps to reproduce it
- Expected vs. actual behaviour
- Your Java version (`java -version`) and operating system

## Suggesting Improvements

Have an idea to improve NEFUSI? [Open a GitHub Issue](https://github.com/jorge-martinez-gil/nefusi/issues/new) with the label **`enhancement`** and describe:

- The improvement you have in mind
- Why it would be valuable
- Any relevant references or examples

## Submitting Code Changes

1. **Fork** the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes.** Ensure the build still passes:
   ```bash
   mvn package -DskipTests
   ```
3. **Open a Pull Request** against the `main` branch with a clear description of what changed and why.

> **Important:** Do not modify `src/nefusi/nefusi.java` or `src/nefusi/Pair.java` unless the change is directly related to a bug fix or the improvement being proposed.

## Code Style

- Follow the existing code conventions (indentation, naming, formatting).
- Add Javadoc comments to all new `public` methods and classes.
- Keep changes focused — one feature or fix per pull request.
