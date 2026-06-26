import pandas as pd

from src.data.load_data import _clean_text_value, clean_raw_dataframe


def test_clean_text_value_strips_quotes_whitespace_and_empty_values():
    assert _clean_text_value("   'YES'   ") == "YES"
    assert pd.isna(_clean_text_value("  "))
    assert pd.isna(_clean_text_value("  ' '  "))
    assert _clean_text_value(15) == 15


def test_clean_raw_dataframe_cleans_text_values_and_maps_target_column():
    test_df = pd.DataFrame(
        {
            "firstcolumn": [1, "?", "'Middle Eastern '"],
            "Class/ASD": ["YES", "NO", "YES"],
        }
    )

    cleaned = clean_raw_dataframe(test_df)

    assert cleaned["firstcolumn"].iloc[0] == 1
    assert pd.isna(cleaned["firstcolumn"].iloc[1])
    assert cleaned["firstcolumn"].iloc[2] == "Middle Eastern"
    assert cleaned["Class/ASD"].tolist() == [1, 0, 1]
    

def test_clean_raw_dataframe_raises_error_for_unexpected_target_values():
    test_df = pd.DataFrame(
        {
            "firstcolumn": [1, 2, 3],
            "Class/ASD": ["YES", "NO", "MAYBE"],
        }
    )

    try:
        clean_raw_dataframe(test_df)
    except ValueError as error:
        assert "Unexpected target values" in str(error)
        assert "MAYBE" in str(error)
    else:
        raise AssertionError("Expected ValueError for unexpected target values")