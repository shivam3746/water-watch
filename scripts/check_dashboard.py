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
    public = not args.url.startswith(("http://127.0.0.1", "http://localhost"))
    output = root / "artifacts/dashboard_checks" / ("public" if public else "local")
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
            surface = page
            if public:
                page.locator("iframe").first.wait_for(timeout=60000)
                surface = page.query_selector("iframe").content_frame()
                assert surface is not None
            surface.get_by_test_id("stPlotlyChart").first.wait_for(timeout=60000)
            surface.wait_for_function("document.querySelectorAll('.js-plotly-plot .js-line').length >= 2")
            assert surface.get_by_test_id("stException").count() == 0
            page.screenshot(path=str(output / f"{name}_replay.png"), full_page=True)
            surface.get_by_test_id("stPlotlyChart").first.scroll_into_view_if_needed()
            page.screenshot(path=str(output / f"{name}_chart.png"))
            surface.get_by_role("tab", name="Detector Comparison", exact=True).click()
            surface.get_by_text("Across all development months", exact=True).wait_for()
            page.screenshot(path=str(output / f"{name}_comparison.png"), full_page=True)
            surface.get_by_role("tab", name="Event Evidence", exact=True).click()
            surface.get_by_text("Completed-month event audit", exact=True).wait_for()
            assert surface.get_by_test_id("stDataFrame").count() >= 1
            surface.get_by_role("tab", name="Historical Replay", exact=True).click()
            window = surface.get_by_test_id("stSelectbox").filter(has=surface.get_by_text("Event window", exact=True))
            combo = window.get_by_role("combobox")
            if combo.evaluate("element => element.tagName") == "SELECT":
                combo.select_option(index=1)
            else:
                combo.focus()
                combo.press("ArrowDown")
                combo.press("ArrowDown")
                combo.press("Enter")
            page.wait_for_timeout(700)
            assert surface.get_by_test_id("stException").count() == 0
            assert not errors, errors
            surface.get_by_role("tab", name="Incident Review", exact=True).click()
            surface.get_by_text("Evidence-backed incident review", exact=True).wait_for()
            if public:
                start = surface.get_by_role("button", name="Start human review", exact=False)
                start.wait_for(timeout=30000)
                if name == "desktop":
                    start.click()
                    surface.get_by_role("textbox", name="Reviewer alias (demo)", exact=True).fill("deployment-check")
                    surface.get_by_text("Reject review recommendation", exact=True).click()
                    assert surface.get_by_role("radio", name="Reject review recommendation", exact=True).is_checked()
                    surface.get_by_role("textbox", name="Decision rationale", exact=True).fill("Synthetic deployment smoke test; no operator action.")
                    surface.get_by_role("button", name="Record decision", exact=False).click()
                    surface.get_by_text("Decision recorded: reject", exact=False).wait_for(timeout=30000)
                else:
                    assert start.is_visible(), "Another anonymous session's review leaked into this session"
            page.wait_for_timeout(400)
            page.screenshot(path=str(output / f"{name}_incident.png"), full_page=True)
            surface.get_by_role("tab", name="2019 Final Test", exact=True).click()
            surface.get_by_text("Held-out 2019 test", exact=True).wait_for()
            page.wait_for_timeout(400)
            page.screenshot(path=str(output / f"{name}_final_test.png"), full_page=True)
            diagnostics = surface.get_by_text("Post-test failure analysis", exact=True)
            if diagnostics.count():
                diagnostics.scroll_into_view_if_needed()
                final_panel = surface.get_by_role("tabpanel", name="2019 Final Test", exact=True)
                size_plot = final_panel.get_by_test_id("stPlotlyChart").first
                size_plot.wait_for()
                assert size_plot.locator(".scatterlayer .point").count() == 19
                page.screenshot(path=str(output / f"{name}_diagnostics.png"))
                assert surface.get_by_text("Research hypothesis and next experiment", exact=True).count() == 1
                assert surface.get_by_test_id("stException").count() == 0
            overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 2")
            assert not overflow, f"Document overflow at {name} viewport"
            assert not surface.evaluate("document.documentElement.scrollWidth > window.innerWidth + 2"), f"App overflow at {name} viewport"
            print(f"{name}: charts rendered, tabs/event window interactive, no page errors or document overflow")
            page.close()
        browser.close()
    print(f"Screenshots: {output}")


if __name__ == "__main__":
    main()
