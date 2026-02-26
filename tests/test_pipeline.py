import os
from pathlib import Path
import pandas as pd
from unittest.mock import patch, MagicMock

import pipeline

def test_pipeline_smoke(tmp_path):
    """
    Smoke test to verify that the end-to-end pipeline runs without crashing,
    provided we mock the data ingestion step to supply valid dummy data.
    """
    # 5 dummy tickers
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    
    # Create valid synthetic prices (at least 31 rows, 5 assets)
    dates = pd.date_range("2021-01-01", periods=50, freq="B")
    data = 100 + pd.DataFrame(
        data=1 * __import__('numpy').random.randn(50, 5),
        index=dates,
        columns=pd.MultiIndex.from_product([tickers, ["Close"]])
    ).cumsum() # random walk
    
    raw_df = data
    
    # Mock data download and config to point to temp dir
    with patch("pipeline.descargar_lotes", return_value=(raw_df, [])):
        with patch("pipeline.cargar_tickers", return_value=tickers):
            with patch("pipeline.REPORTS_DIR", tmp_path):
                # Run the pipeline
                report_path = pipeline.pipeline_trimestral()
                
    # Verify report was generated
    assert report_path is not None
    assert os.path.exists(report_path)
    
    # Verify CSVs are side-by-side
    dirname = os.path.dirname(report_path)
    csv_files = os.listdir(dirname)
    assert "efficient_frontier.csv" in csv_files
    assert "weights_best.csv" in csv_files
    assert "quality_report.csv" in csv_files
