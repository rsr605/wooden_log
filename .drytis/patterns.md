# Patterns — Wooden Log Detection with YOLOv8

## Coding Standards
- Python 3.8+ compatible
- PEP 8 style (line length 100)
- Type hints on all public functions
- Docstrings on all public classes and functions (Google style)

## Naming Conventions
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

## Error Handling
- Catch specific exceptions, never bare `except:`
- Return meaningful error messages to the user
- Flask routes return proper HTTP status codes
- Log errors with context (what image, what operation)

## Project Conventions
- All classes accept config via constructor or method parameters
- Default confidence threshold: 0.25 (configurable)
- Default IoU threshold: 0.7 (configurable)
- Model weights stored in `models/`
- Uploaded files validated by extension and content type

## Test Conventions
- Test files: `tests/test_*.py`
- Test functions: `test_<behavior>` (snake_case, descriptive)
- Use `unittest` or `pytest` framework
- Each test tests ONE behavior
- Test data generated programmatically (no external fixtures needed)
- Integration tests use Flask test client
- Mock YOLOv8 model loading in unit tests; use real model in smoke tests only

## File Validation (Security)
- Allowed image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- Allowed video extensions: `.mp4`, `.avi`, `.mov`, `.mkv`
- Max upload size: 50 MB
- Sanitize filenames (secure_filename)
