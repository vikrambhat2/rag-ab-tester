import pytest

from src.evaluator.stats import cohens_d, compare_metric, confidence_interval, effect_size_label


def test_clear_challenger_win():
    control = [0.50, 0.55, 0.52, 0.48, 0.51, 0.53, 0.49, 0.50]
    challenger = [0.80, 0.82, 0.79, 0.85, 0.81, 0.83, 0.78, 0.84]
    result = compare_metric("faithfulness", control, challenger)
    assert result.winner == "challenger"
    assert result.significant and result.meaningful
    assert result.delta > 0
    assert result.ci_low > 0


def test_clear_control_win():
    result = compare_metric("m", [0.9, 0.8, 0.85, 0.9, 0.88], [0.2, 0.3, 0.25, 0.3, 0.28])
    assert result.winner == "control"
    assert result.delta < 0


def test_noise_is_no_difference():
    control = [0.5, 0.6, 0.4, 0.55, 0.45, 0.5]
    challenger = [0.52, 0.58, 0.41, 0.54, 0.47, 0.49]
    assert compare_metric("m", control, challenger).winner == "no difference"


def test_identical_scores_do_not_produce_nan():
    result = compare_metric("m", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert result.p_value == 1.0
    assert result.winner == "no difference"
    assert result.delta == 0


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="same length"):
        compare_metric("m", [0.1, 0.2], [0.1])


def test_too_few_samples_rejected():
    with pytest.raises(ValueError, match="At least 2"):
        compare_metric("m", [0.1], [0.2])


def test_cohens_d_sign_and_zero_variance():
    assert cohens_d([0.1, 0.2, 0.3], [0.6, 0.7, 0.8]) > 0
    assert cohens_d([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_confidence_interval_contains_mean_diff():
    low, high = confidence_interval([0.1, 0.2, 0.3, 0.4], [0.2, 0.35, 0.3, 0.5])
    assert low < 0.075 < high


@pytest.mark.parametrize(
    ("d", "label"),
    [(0.0, "negligible"), (0.19, "negligible"), (-0.3, "small"), (0.5, "medium"), (-0.8, "large")],
)
def test_effect_size_label(d, label):
    assert effect_size_label(d) == label
