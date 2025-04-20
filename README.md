# Documentation Vector Database Builder

A tool for building semantic and lexical indexes for documentation of Python open-source projects like scikit-learn, skrub, and skore.

## Overview

This project creates vector databases from documentation of Python libraries to enable:

- Semantic search across documentation
- Efficient retrieval of relevant documentation sections
- Support for both semantic (meaning-based) and lexical (keyword-based) search

## Features

- Documentation scraping from multiple sources
- Text chunking and preprocessing
- Vector embedding generation
- Indexing with multiple strategies
- Search API for querying the indexes

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- [Pixi](https://pixi.sh/) package manager

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd docs-vectordb-builder

# Set up the environment with pixi
pixi install

# Set up pre-commit hooks
pip install pre-commit
pre-commit install
```

### Development Workflow

This project uses pre-commit hooks to enforce code quality standards:

- **Ruff**: For linting, code formatting, and import sorting
- **MyPy**: For static type checking
- **Additional hooks**: For detecting common issues (trailing whitespace, etc.)

To manually run the pre-commit hooks on all files:

```bash
pre-commit run --all-files
```

## Usage

*To be added as the project develops*

## License

[MIT](LICENSE)
