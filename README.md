[![CI](https://github.com/joaquinorradre/MLOps-Lab1/actions/workflows/CI.yml/badge.svg)](https://github.com/joaquinorradre/MLOps-Lab1/actions/workflows/CI.yml)
# MLOps-Lab1

## Project Overview

This project provides a minimal end-to-end machine learning application together with a fully automated Continuous Integration pipeline. It includes a small core library with basic image-processing functions, like resizing or predicting the class (in this case randomly), a command-line interface built with Click, and a FastAPI web service that exposes the same functionality through HTTP endpoints. The repository also contains a complete pytest suite with full test coverage. All development workflows are automated through GitHub Actions, which run code formatting, linting, dependency installation, and the entire test suite on every push and pull request.

## Project Structure

```
MLOps-Lab1/
├── .github/
│   └── workflows/
│       └── CI.yml        # CI Pipeline definition
├── api/
│   └── api.py     # FastAPI application
├── cli/
│   └── cli.py              # Click command-line interface
├── mylib/
│   └── logic.py            # Core project logic
├── templates/
│   └── home.html           # API homepage
├── tests/
│   ├── test_api.py         # Tests for the API
│   ├── test_cli.py         # Tests for the CLI
│   └── test_logic.py       # Tests for the logic
├── .gitignore
├── LICENSE
├── Makefile                # Make commands for quality checks
├── pyproject.toml          # Project metadata and dependencies
└── uv.lock                 # Pinned dependency versions
```

## Setup and Installation
This project uses uv for package management.

1. **Clone the repository**:

    ```bash
       git clone https://github.com/joaquinorradre/MLOps-Lab1.git
       cd MLOps
    ```
2. **Install dependencies**

    ```bash
        make install
    ```
3. **Activate the new environment**

    On Windows (PowerShell)
    ```bash
    .\.venv\Scripts\Activate.ps1
     ```
    On macOS/Linux (Bash)
    ```bash
    source .venv/Scripts/activate
    ```

## Usage

### Quality checks (Makefile)

The Makefile contains all commands for code quality.

```bash
  # Auto-format all code with black.
  make format

  # Lint all code with pylint.
  make lint

  # Run all tests with pytest and generate a coverage report.
  make test

  # Run all local checks (format, lint, test).
  make all
  ```

### Running the API
To start the web server:

```bash
uv run python -m api.api
```

You can now access:

- API Homepage: http://localhost:8000

- API Docs (to see the endpoints): http://localhost:8000/docs

#### API Endpoints

1. **POST /predict**
   <img width="1423" height="924" alt="image" src="https://github.com/user-attachments/assets/054b2a28-c014-4e26-953e-c5d39e85057b" />

    - Description: Takes an image file and returns a (mocked) string prediction.
    - Input: multipart/form-data
    - File: The image file (UploadFile).
    - Output (JSON):
    ```bash
    {
      "filename": "my_image.jpg",
      "prediction": "dog"
    }
    ```

2. **POST /resize**
   <img width="1080" height="906" alt="image" src="https://github.com/user-attachments/assets/8ba332eb-c97c-467e-97e8-eba7f5840c78" />


    - Description: Takes an image file and target dimensions, returns the resized image.
    - Input: multipart/form-data
    - File: The image file (UploadFile).
    - Width: The target width (int form data).
    - Height: The target height (int form data).
    - Output: The resized image file (image/png).

3. **POST /resize**
    <img width="1344" height="898" alt="image" src="https://github.com/user-attachments/assets/c090764a-e844-46a9-b4b3-bdf7b466ce40" />


    - Takes an image file and returns a grayscale version of it.
    - Input: multipart/form-data
    - File: The image file (UploadFile).
    - Output: The processed grayscale image file (image/png

### Running the CLI

You can also use the logic via the command-line interface.

Predict Example:

```bash
uv run python -m cli.cli predict path/to/your/image.jpg
```

Output: Prediction: cat

Resize Example:

```bash
uv run python -m cli.cli resize "path/to/image.jpg" --width 50 --height 50 "output/resized.png"
```

Output: Image resized and saved to: output/resized.png









