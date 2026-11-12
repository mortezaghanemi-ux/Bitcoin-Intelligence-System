# Candlestick Pattern Analysis

This module detects and analyzes Bitcoin candlestick patterns.

Pattern categories:
- Doji patterns
- Hammer and Hanging Man
- Engulfing patterns
- Marubozu
- Pin Bar
- Tweezer Top and Bottom
- Morning Star and Evening Star
- Three White Soldiers
- Three Black Crows
- Rising Three Methods
- Falling Three Methods
- Harami patterns
- Other recognized candlestick patterns

Supported timeframes:
- 5m
- 15m
- 1h
- 4h

Rules:
- Analyze closed candles only.
- Never use future candle information.
- Record pattern direction and confidence.
- Include detection timestamp in UTC.
- Avoid duplicate pattern records.
- Preserve historical detections.