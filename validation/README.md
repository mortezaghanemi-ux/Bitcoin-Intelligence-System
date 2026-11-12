# Data Validation

This module validates all collected data.

Checks:
- Missing values
- Invalid timestamps
- Duplicate records
- Stale data
- Invalid prices and volumes
- Data consistency

Rules:
- All timestamps must use UTC.
- Invalid data must be rejected.
- Validation results must be recorded.
- Dependent analysis must stop when required data is invalid.