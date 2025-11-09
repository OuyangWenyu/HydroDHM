# Documentation

This directory contains the HydroDHM documentation built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

## Building Documentation Locally

### Prerequisites

```bash
# Install documentation dependencies
uv sync --extra docs

# Or with pip
pip install mkdocs mkdocs-material mkdocs-git-revision-date-localized-plugin
```

### Serve Locally

```bash
# Start development server
uv run mkdocs serve

# Or with pip
mkdocs serve
```

Open http://localhost:8000 in your browser.

### Build Static Site

```bash
# Build documentation
uv run mkdocs build

# Output will be in site/ directory
```

## Documentation Structure

```
docs/
├── index.md                    # Home page
├── getting-started/            # Installation and quick start
│   ├── installation.md
│   ├── quickstart.md
│   └── configuration.md
├── models/                     # Model documentation
│   ├── overview.md
│   ├── xaj/                    # XAJ model
│   └── deep-learning/          # LSTM and DPL-XAJ
├── datasets/                   # Dataset guides
├── tutorials/                  # Step-by-step tutorials
├── advanced/                   # Advanced topics
├── api/                        # API reference
└── about/                      # Citation, contributing, license
```

## Contributing to Documentation

### Adding a New Page

1. Create a new `.md` file in the appropriate directory
2. Add it to `mkdocs.yml` navigation
3. Use Markdown with Material for MkDocs extensions

### Markdown Features

#### Admonitions

```markdown
!!! note
    This is a note

!!! warning
    This is a warning

!!! tip
    This is a tip
```

#### Code Blocks

```markdown
\`\`\`python
def hello():
    print("Hello, HydroDHM!")
\`\`\`
```

#### Tabs

```markdown
=== "Python"
    \`\`\`python
    print("Hello")
    \`\`\`

=== "Bash"
    \`\`\`bash
    echo "Hello"
    \`\`\`
```

## Automatic Deployment

Documentation is automatically deployed to GitHub Pages when pushing to the `main` branch via GitHub Actions (`.github/workflows/docs.yml`).

View live documentation at: https://OuyangWenyu.github.io/HydroDHM

## Style Guide

- Use clear, concise language
- Include code examples
- Add screenshots where helpful
- Keep line length reasonable (~80-100 characters)
- Use proper headings hierarchy (H1 → H2 → H3)
- Link to related documentation

## Questions?

Open an issue on [GitHub](https://github.com/OuyangWenyu/HydroDHM/issues) or see [Contributing Guide](about/contributing.md).
