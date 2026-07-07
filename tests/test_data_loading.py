import pandas as pd

from src.data.load_data import _clean_text_value, clean_raw_dataframe


def test_clean_text_value_normalizes_quoted_or_empty_strings():
    assert _clean_text_value("   'YES'   ") == "YES"
    assert pd.isna(_clean_text_value("  "))
    assert pd.isna(_clean_text_value("  ' '  "))
    assert _clean_text_value(15) == 15


def test_clean_raw_dataframe_normalizes_text_columns_and_maps_target_labels():
    test_df = pd.DataFrame(
        {
            "country_of_res": ["Turkey", "?", "'United Kingdom '"],
            "Class/ASD": ["YES", "NO", "YES"],
        }
    )

    cleaned = clean_raw_dataframe(test_df)

    assert cleaned["country_of_res"].iloc[0] == "Turkey"
    assert pd.isna(cleaned["country_of_res"].iloc[1])
    assert cleaned["country_of_res"].iloc[2] == "United Kingdom"
    assert cleaned["Class/ASD"].tolist() == [1, 0, 1]


def test_clean_raw_dataframe_raises_error_for_unexpected_target_values():
    test_df = pd.DataFrame(
        {
            "age": [10, 12, 14],
            "Class/ASD": ["YES", "NO", "MAYBE"],
        }
    )

    try:
        clean_raw_dataframe(test_df)
    except ValueError as error:
        assert "Unexpected target values" in str(error)
        assert "MAYBE" in str(error)
        assert "Expected target values" in str(error)
        assert "NO, YES" in str(error)
    else:
        raise AssertionError("Expected ValueError for unexpected target values")