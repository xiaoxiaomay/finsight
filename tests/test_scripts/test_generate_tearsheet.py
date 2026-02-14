"""Tests for tearsheet generator script."""

from scripts.generate_tearsheet import _calc, _generate_data, generate_tearsheet_html


class TestGenerateData:
    def test_returns_dict(self):
        data = _generate_data()
        assert isinstance(data, dict)

    def test_has_required_keys(self):
        data = _generate_data()
        required = {"port_ret", "bench_ret", "port_val", "bench_val",
                    "is_ret", "oos_ret", "factors", "attribution", "live_start"}
        assert required.issubset(set(data.keys()))

    def test_full_length(self):
        data = _generate_data()
        assert len(data["port_ret"]) == 1008 + 504

    def test_factors_count(self):
        data = _generate_data()
        assert len(data["factors"]) == 10


class TestCalc:
    def test_returns_dict(self):
        data = _generate_data()
        m = _calc(data["is_ret"], data["is_ret"])
        assert isinstance(m, dict)

    def test_has_sharpe(self):
        data = _generate_data()
        m = _calc(data["is_ret"], data["is_ret"])
        assert "sharpe" in m


class TestGenerateTearsheetHtml:
    def test_returns_html_string(self):
        html = generate_tearsheet_html()
        assert isinstance(html, str)
        assert "<html>" in html

    def test_contains_strategy_name(self):
        html = generate_tearsheet_html(strategy_name="Test Strategy")
        assert "Test Strategy" in html

    def test_contains_key_sections(self):
        html = generate_tearsheet_html()
        assert "Key Performance Metrics" in html
        assert "Equity Curve" in html
        assert "Drawdown" in html
        assert "Monthly Returns" in html
        assert "Factor Attribution" in html

    def test_contains_email_when_provided(self):
        html = generate_tearsheet_html(email="test@example.com")
        assert "test@example.com" in html

    def test_no_email_when_not_provided(self):
        html = generate_tearsheet_html()
        assert "Contact:" not in html

    def test_contains_disclaimer(self):
        html = generate_tearsheet_html()
        assert "Disclaimer" in html

    def test_dark_theme_styling(self):
        html = generate_tearsheet_html()
        assert "#0d1117" in html  # Dark background
