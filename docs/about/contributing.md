# Contributing

We welcome contributions to HydroDHM! This guide will help you get started.

## Ways to Contribute

### 1. Report Bugs

Found a bug? Please [open an issue](https://github.com/OuyangWenyu/HydroDHM/issues) with:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)

### 2. Suggest Features

Have an idea? Open an issue with:

- Use case description
- Proposed solution
- Alternative approaches considered

### 3. Improve Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples
- Translate to other languages
- Expand tutorials

### 4. Submit Code

Ready to code? Follow these steps:

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/HydroDHM.git
cd HydroDHM
```

### 2. Create Development Environment

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
uv sync --all-extras
```

### 3. Create a Branch

```bash
git checkout -b feature/my-new-feature
# or
git checkout -b fix/bug-description
```

## Making Changes

### Code Style

We follow PEP 8 and use:

- **Black** for formatting
- **isort** for import sorting
- **flake8** for linting

```bash
# Format code
uv run black hydrodhm/
uv run isort hydrodhm/

# Check linting
uv run flake8 hydrodhm/
```

### Documentation

Update documentation for any new features:

```bash
# Edit docs in docs/
vim docs/models/new-model.md

# Preview locally
uv run mkdocs serve
# Open http://localhost:8000
```

### Testing

Add tests for new features:

```bash
# Run tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_xaj.py -v

# Check coverage
uv run pytest --cov=hydrodhm tests/
```

## Submitting Changes

### 1. Commit Your Changes

Write clear commit messages:

```bash
git add .
git commit -m "Add feature: description of feature

- Detailed point 1
- Detailed point 2

Fixes #123"
```

### 2. Push to Your Fork

```bash
git push origin feature/my-new-feature
```

### 3. Create Pull Request

- Go to GitHub and create a Pull Request
- Fill in the PR template
- Link related issues
- Wait for review

## Pull Request Guidelines

### Good PR Checklist

- [ ] Code follows project style
- [ ] Tests added for new features
- [ ] Documentation updated
- [ ] All tests pass
- [ ] Clear description of changes
- [ ] Linked to relevant issues

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
How was this tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
```

## Code Review Process

1. **Maintainer Review**: A maintainer will review your PR
2. **Feedback**: Address any comments or requests
3. **Approval**: Once approved, maintainer will merge
4. **Recognition**: You'll be added to contributors!

## Development Guidelines

### Adding a New Model

1. Create model class in `hydrodhm/models/`
2. Add configuration in `hydrodhm/configs/`
3. Add tests in `tests/models/`
4. Document in `docs/models/`
5. Add example in `examples/`

### Adding a New Dataset

1. Extend dataset class in `hydrodhm/datasets/`
2. Add data loader
3. Add tests
4. Document data format
5. Provide example

## Community Guidelines

### Be Respectful

- Use welcoming language
- Respect differing viewpoints
- Accept constructive criticism
- Focus on what's best for the community

### Get Help

- **Questions**: GitHub Discussions
- **Bugs**: GitHub Issues
- **Chat**: [Join our community](#)

## Recognition

Contributors are recognized in:

- README contributors section
- Release notes
- Documentation credits

## License

By contributing, you agree that your contributions will be licensed under the BSD License.

## Questions?

Feel free to open an issue or reach out to the maintainers!

**Happy Contributing! 🎉**
