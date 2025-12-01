#!/usr/bin/env python
"""
Dashboard Validation Script

Uses Playwright to validate the Crypto Anomaly Detection Dashboard.
Validates all pages and takes screenshots at each step.

Usage:
    # Start dashboard first:
    python deployments/dashboard/app.py

    # Then run validation:
    python scripts/validate_dashboard.py

    # Run with visible browser:
    python scripts/validate_dashboard.py --headed

    # Run with custom branch:
    python scripts/validate_dashboard.py --branch user.alice@example.com
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from playwright.async_api import async_playwright, Page, Browser


# Configuration
BASE_URL = "http://localhost:8001"
DEFAULT_BRANCH = "user_eddie_at_outerbounds_co"  # Sanitized branch name
SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"


@dataclass
class ValidationResult:
    """Container for validation results."""
    page_name: str
    passed: bool
    checks: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    screenshot_path: Optional[str] = None


class DashboardValidator:
    """Validates the Crypto Anomaly Detection Dashboard."""

    def __init__(self, branch: str, headed: bool = False):
        self.branch = branch
        self.headed = headed
        self.results: list[ValidationResult] = []
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    async def setup(self):
        """Initialize browser and page."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=not self.headed)
        self.page = await self.browser.new_page()

        # Set viewport for consistent screenshots
        await self.page.set_viewport_size({"width": 1280, "height": 900})

        # Ensure screenshots directory exists
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    async def teardown(self):
        """Clean up browser resources."""
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()

    async def take_screenshot(self, name: str) -> str:
        """Take a screenshot and return the path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{name}.png"
        path = SCREENSHOTS_DIR / filename
        await self.page.screenshot(path=str(path), full_page=True)
        return str(path)

    async def navigate_and_set_branch(self):
        """Navigate to dashboard and set the branch."""
        print(f"\n{'='*60}")
        print("NAVIGATING TO DASHBOARD")
        print(f"{'='*60}")

        # Navigate to root
        await self.page.goto(f"{BASE_URL}/?branch={self.branch}")
        await self.page.wait_for_load_state("networkidle")

        # Verify branch is set in input
        branch_input = self.page.locator("#branch-input")
        current_branch = await branch_input.input_value()
        print(f"Branch set to: {current_branch}")

        return current_branch == self.branch

    async def validate_overview(self) -> ValidationResult:
        """Validate the Overview page."""
        print(f"\n{'='*60}")
        print("VALIDATING OVERVIEW PAGE")
        print(f"{'='*60}")

        result = ValidationResult(page_name="Overview", passed=True)

        try:
            # Navigate to overview
            await self.page.goto(f"{BASE_URL}/?branch={self.branch}")
            await self.page.wait_for_load_state("networkidle")

            # Check for main elements
            checks = {}

            # Check page title
            title = await self.page.title()
            checks["title"] = "Crypto Anomaly Detection" in title
            print(f"  Title: {title} - {'PASS' if checks['title'] else 'FAIL'}")

            # Check for pipeline status card
            pipeline_card = self.page.locator(".card").first
            checks["pipeline_card_exists"] = await pipeline_card.is_visible()
            print(f"  Pipeline card visible: {'PASS' if checks['pipeline_card_exists'] else 'FAIL'}")

            # Check for stats (look for stat values)
            stat_values = self.page.locator(".stat-value")
            stat_count = await stat_values.count()
            checks["has_stats"] = stat_count > 0
            print(f"  Stats displayed ({stat_count}): {'PASS' if checks['has_stats'] else 'FAIL'}")

            # Check for badges (status indicators)
            badges = self.page.locator(".badge")
            badge_count = await badges.count()
            checks["has_badges"] = badge_count > 0
            print(f"  Status badges ({badge_count}): {'PASS' if checks['has_badges'] else 'FAIL'}")

            # Look for champion model info
            page_content = await self.page.content()
            checks["shows_champion"] = "champion" in page_content.lower()
            print(f"  Champion info present: {'PASS' if checks['shows_champion'] else 'FAIL'}")

            # Look for evaluation info
            checks["shows_evaluation"] = "evaluat" in page_content.lower()
            print(f"  Evaluation info present: {'PASS' if checks['shows_evaluation'] else 'FAIL'}")

            result.checks = checks
            result.passed = all(checks.values())

            # Take screenshot
            result.screenshot_path = await self.take_screenshot("01_overview")
            print(f"  Screenshot: {result.screenshot_path}")

        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            print(f"  ERROR: {e}")

        return result

    async def validate_data_page(self) -> ValidationResult:
        """Validate the Data Explorer page."""
        print(f"\n{'='*60}")
        print("VALIDATING DATA PAGE")
        print(f"{'='*60}")

        result = ValidationResult(page_name="Data", passed=True)

        try:
            # Navigate to data page
            await self.page.goto(f"{BASE_URL}/data?branch={self.branch}")
            await self.page.wait_for_load_state("networkidle")

            checks = {}

            # Check for data assets section
            page_content = await self.page.content()

            # Check for market_snapshot
            checks["has_market_snapshot"] = "market_snapshot" in page_content
            print(f"  market_snapshot asset: {'PASS' if checks['has_market_snapshot'] else 'FAIL'}")

            # Check for training_dataset
            checks["has_training_dataset"] = "training_dataset" in page_content
            print(f"  training_dataset asset: {'PASS' if checks['has_training_dataset'] else 'FAIL'}")

            # Check for eval_holdout
            checks["has_eval_holdout"] = "eval_holdout" in page_content
            print(f"  eval_holdout asset: {'PASS' if checks['has_eval_holdout'] else 'FAIL'}")

            # Check for tables
            tables = self.page.locator("table")
            table_count = await tables.count()
            checks["has_tables"] = table_count > 0
            print(f"  Data tables ({table_count}): {'PASS' if checks['has_tables'] else 'FAIL'}")

            # Check for sample counts
            checks["shows_samples"] = "sample" in page_content.lower() or "100" in page_content
            print(f"  Sample counts shown: {'PASS' if checks['shows_samples'] else 'FAIL'}")

            result.checks = checks
            result.passed = all(checks.values())

            # Take screenshot
            result.screenshot_path = await self.take_screenshot("02_data")
            print(f"  Screenshot: {result.screenshot_path}")

        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            print(f"  ERROR: {e}")

        return result

    async def validate_models_page(self) -> ValidationResult:
        """Validate the Model Registry page."""
        print(f"\n{'='*60}")
        print("VALIDATING MODELS PAGE")
        print(f"{'='*60}")

        result = ValidationResult(page_name="Models", passed=True)

        try:
            # Navigate to models page
            await self.page.goto(f"{BASE_URL}/models?branch={self.branch}")
            await self.page.wait_for_load_state("networkidle")

            checks = {}
            page_content = await self.page.content()

            # Check for model versions table
            tables = self.page.locator("table")
            table_count = await tables.count()
            checks["has_versions_table"] = table_count > 0
            print(f"  Versions table: {'PASS' if checks['has_versions_table'] else 'FAIL'}")

            # Check for status badges
            badges = self.page.locator(".badge")
            badge_count = await badges.count()
            checks["has_status_badges"] = badge_count > 0
            print(f"  Status badges ({badge_count}): {'PASS' if checks['has_status_badges'] else 'FAIL'}")

            # Check for champion badge specifically
            checks["has_champion_badge"] = "champion" in page_content.lower()
            print(f"  Champion badge: {'PASS' if checks['has_champion_badge'] else 'FAIL'}")

            # Check for algorithm info
            checks["shows_algorithm"] = "isolation_forest" in page_content.lower()
            print(f"  Algorithm shown: {'PASS' if checks['shows_algorithm'] else 'FAIL'}")

            # Check for anomaly rate
            checks["shows_anomaly_rate"] = "anomaly" in page_content.lower() or "10%" in page_content
            print(f"  Anomaly rate shown: {'PASS' if checks['shows_anomaly_rate'] else 'FAIL'}")

            # Check for compare functionality
            compare_buttons = self.page.locator("button:has-text('Compare'), a:has-text('Compare')")
            compare_count = await compare_buttons.count()
            checks["has_compare"] = compare_count > 0 or "compare" in page_content.lower()
            print(f"  Compare functionality: {'PASS' if checks['has_compare'] else 'FAIL'}")

            result.checks = checks
            result.passed = all(checks.values())

            # Take screenshot
            result.screenshot_path = await self.take_screenshot("03_models")
            print(f"  Screenshot: {result.screenshot_path}")

        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            print(f"  ERROR: {e}")

        return result

    async def validate_navigation(self) -> ValidationResult:
        """Validate navigation between pages."""
        print(f"\n{'='*60}")
        print("VALIDATING NAVIGATION")
        print(f"{'='*60}")

        result = ValidationResult(page_name="Navigation", passed=True)

        try:
            # Start from overview
            await self.page.goto(f"{BASE_URL}/?branch={self.branch}")
            await self.page.wait_for_load_state("networkidle")

            checks = {}

            # Click Data link
            await self.page.click("a[href*='/data']")
            await self.page.wait_for_load_state("networkidle")
            checks["nav_to_data"] = "/data" in self.page.url
            print(f"  Navigate to Data: {'PASS' if checks['nav_to_data'] else 'FAIL'}")

            # Click Models link
            await self.page.click("a[href*='/models']")
            await self.page.wait_for_load_state("networkidle")
            checks["nav_to_models"] = "/models" in self.page.url
            print(f"  Navigate to Models: {'PASS' if checks['nav_to_models'] else 'FAIL'}")

            # Click Overview link
            await self.page.click("a[href*='/?branch']")
            await self.page.wait_for_load_state("networkidle")
            checks["nav_to_overview"] = self.page.url.endswith(f"/?branch={self.branch}") or "/?branch=" in self.page.url
            print(f"  Navigate to Overview: {'PASS' if checks['nav_to_overview'] else 'FAIL'}")

            # Verify branch persists across navigation
            branch_input = self.page.locator("#branch-input")
            current_branch = await branch_input.input_value()
            checks["branch_persists"] = current_branch == self.branch
            print(f"  Branch persists: {'PASS' if checks['branch_persists'] else 'FAIL'}")

            result.checks = checks
            result.passed = all(checks.values())

        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            print(f"  ERROR: {e}")

        return result

    async def validate_api_health(self) -> ValidationResult:
        """Validate API health endpoint."""
        print(f"\n{'='*60}")
        print("VALIDATING API HEALTH")
        print(f"{'='*60}")

        result = ValidationResult(page_name="API Health", passed=True)

        try:
            response = await self.page.goto(f"{BASE_URL}/health")

            checks = {}
            checks["status_200"] = response.status == 200
            print(f"  HTTP 200: {'PASS' if checks['status_200'] else 'FAIL'}")

            # Check response body
            body = await response.json()
            checks["healthy_status"] = body.get("status") == "healthy"
            print(f"  Healthy status: {'PASS' if checks['healthy_status'] else 'FAIL'}")

            result.checks = checks
            result.passed = all(checks.values())

        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            print(f"  ERROR: {e}")

        return result

    async def run_validation(self) -> bool:
        """Run all validation checks."""
        print(f"\n{'#'*60}")
        print("CRYPTO ANOMALY DETECTION DASHBOARD VALIDATION")
        print(f"{'#'*60}")
        print(f"Dashboard URL: {BASE_URL}")
        print(f"Branch: {self.branch}")
        print(f"Headed mode: {self.headed}")

        try:
            await self.setup()

            # Run all validations
            self.results.append(await self.validate_api_health())

            if not await self.navigate_and_set_branch():
                print(f"WARNING: Branch mismatch - expected {self.branch}")

            self.results.append(await self.validate_overview())
            self.results.append(await self.validate_data_page())
            self.results.append(await self.validate_models_page())
            self.results.append(await self.validate_navigation())

            # Print summary
            self.print_summary()

            return all(r.passed for r in self.results)

        finally:
            await self.teardown()

    def print_summary(self):
        """Print validation summary."""
        print(f"\n{'#'*60}")
        print("VALIDATION SUMMARY")
        print(f"{'#'*60}")

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.page_name}")
            if result.errors:
                for error in result.errors:
                    print(f"       Error: {error}")

        print(f"\n{'='*60}")
        print(f"RESULT: {passed}/{total} validations passed")
        print(f"{'='*60}")

        if passed == total:
            print("\nAll validations passed!")
        else:
            print(f"\n{total - passed} validation(s) failed.")

        print(f"\nScreenshots saved to: {SCREENSHOTS_DIR}")


async def main():
    parser = argparse.ArgumentParser(description="Validate the Crypto Anomaly Detection Dashboard")
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help="Branch to validate")
    parser.add_argument("--headed", action="store_true", help="Run with visible browser")
    args = parser.parse_args()

    validator = DashboardValidator(branch=args.branch, headed=args.headed)
    success = await validator.run_validation()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
