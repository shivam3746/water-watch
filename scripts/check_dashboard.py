"""Optional browser smoke check; uses an installed Chrome or Edge executable."""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/dashboard_checks"
    output.mkdir(parents=True, exist_ok=True)
    executables = [Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
                   Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")]
    installed = next((str(path) for path in executables if path.exists()), None)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=installed, headless=True)
        for name, size in [("desktop", {"width": 1440, "height": 1000}), ("mobile", {"width": 390, "height": 844})]:
            page = browser.new_page(viewport=size)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(args.url, wait_until="networkidle")
            page.get_by_test_id("stPlotlyChart").first.wait_for(timeout=30000)
            page.wait_for_function("document.querySelectorAll('.js-plotly-plot .js-line').length >= 2")
            assert page.get_by_test_id("stException").count() == 0
            page.screenshot(path=str(output / f"{name}_replay.png"), full_page=True)
            page.get_by_test_id("stPlotlyChart").first.scroll_into_view_if_needed()
            page.screenshot(path=str(output / f"{name}_chart.png"))
            page.get_by_role("tab", name="Detector Comparison", exact=True).click()
            page.get_by_text("Across all development months", exact=True).wait_for()
            page.screenshot(path=str(output / f"{name}_comparison.png"), full_page=True)
            page.get_by_role("tab", name="Event Evidence", exact=True).click()
            page.get_by_text("Completed-month event audit", exact=True).wait_for()
            assert page.get_by_test_id("stDataFrame").count() >= 1
            page.get_by_role("tab", name="Historical Replay", exact=True).click()
            window = page.get_by_test_id("stSelectbox").filter(has=page.get_by_text("Event window", exact=True))
            combo = window.get_by_role("combobox")
            if combo.evaluate("element => element.tagName") == "SELECT":
                combo.select_option(index=1)
            else:
                combo.focus()
                combo.press("ArrowDown")
                combo.press("ArrowDown")
                combo.press("Enter")
            page.wait_for_timeout(700)
            assert page.get_by_test_id("stException").count() == 0
            assert not errors, errors
            page.get_by_role("tab", name="Incident Review", exact=True).click()
            page.get_by_text("Evidence-backed incident review", exact=True).wait_for()
            page.wait_for_timeout(400)
            page.screenshot(path=str(output / f"{name}_incident.png"), full_page=True)
            page.get_by_role("tab", name="2019 Final Test", exact=True).click()
            page.get_by_text("Held-out 2019 test", exact=True).wait_for()
            page.wait_for_timeout(400)
            page.screenshot(path=str(output / f"{name}_final_test.png"), full_page=True)
            overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 2")
            assert not overflow, f"Document overflow at {name} viewport"
            print(f"{name}: charts rendered, tabs/event window interactive, no page errors or document overflow")
            page.close()
        browser.close()
    print(f"Screenshots: {output}")


if __name__ == "__main__":
    main()
