@pytest.mark.parametrize(
    "probabilities",
    [
        None,
        [],
        (),
        "70,20,10",
        100,
    ],
)
def test_forecast_validator_rejects_invalid_probability_container(
    probabilities,
):
    forecast = make_forecast()
    forecast["probabilities"] = probabilities

    with pytest.raises(ForecastEvaluationError):
        _validate_forecast(forecast)