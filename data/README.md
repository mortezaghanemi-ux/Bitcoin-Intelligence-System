# Data Architecture

This directory stores all Bitcoin Intelligence System data.

Directory structure:

- raw/
- processed/
- forecast/
- trader/

Data management rules:

- Raw data must never be overwritten.
- Processed data must pass validation.
- Forecast and Trader data must remain separate.
- All timestamps must use UTC.
- Every dataset must have a defined owner.
- Duplicate records must be prevented.
- Historical records must be preserved.
- Invalid data must never reach analysis engines.
- API keys and secrets must never be stored here.