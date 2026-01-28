# Contributing to Customer Segmentation Project

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)
- Screenshots if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:
- Check if the enhancement has already been suggested
- Provide a clear description of the feature
- Explain why it would be useful
- Include examples if possible

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/customer-segmentation-project.git
   cd customer-segmentation-project
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   pytest tests/
   python src/main.py  # Ensure it runs
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add amazing feature"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

7. **Open a Pull Request**
   - Provide a clear description of your changes
   - Link any related issues
   - Wait for review

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Use type hints where appropriate

### Example:

```python
def calculate_customer_value(
    purchases: pd.DataFrame,
    customer_id: str
) -> float:
    """
    Calculate total value for a specific customer.
    
    Args:
        purchases: DataFrame containing purchase history
        customer_id: Unique identifier for the customer
        
    Returns:
        Total monetary value of customer purchases
    """
    customer_purchases = purchases[purchases['CustomerID'] == customer_id]
    return customer_purchases['TotalAmount'].sum()
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings for all new functions
- Update inline comments as needed
- Include examples in docstrings

### Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage
- Use descriptive test names

### Commit Messages

Use clear, concise commit messages:
- Start with a verb (Add, Fix, Update, Remove)
- Keep first line under 50 characters
- Add detailed description if needed

Good examples:
```
Add customer lifetime value calculation
Fix RFM score calculation for edge cases
Update documentation for dashboard setup
```

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/customer-segmentation-project.git
   cd customer-segmentation-project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run tests**
   ```bash
   pytest tests/ -v
   ```

## Code Review Process

All pull requests will be reviewed by maintainers. We look for:

1. **Code Quality**
   - Clean, readable code
   - Proper error handling
   - Efficient algorithms

2. **Testing**
   - Adequate test coverage
   - All tests passing

3. **Documentation**
   - Clear docstrings
   - Updated README if needed

4. **Functionality**
   - Works as intended
   - No breaking changes

## Questions?

Feel free to open an issue for any questions about contributing!

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help create a welcoming environment

Thank you for contributing! 🎉
