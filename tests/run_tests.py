#!/usr/bin/env python3
"""
Master Test Runner for DataMender
Runs Unit, Integration, and E2E tests and produces a refined grand total.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_unit import UnitTestSuite
from test_integration import IntegrationTestSuite
from test_e2e import DataMenderE2ETest

def main():
    total_passed = 0
    total_count = 0
    
    print("Running Unit Tests...")
    unit = UnitTestSuite()
    unit.run_all_tests()
    
    unit_passed = sum(1 for r in unit.test_results if r["passed"])
    unit_total = len(unit.test_results)
    total_passed += unit_passed
    total_count += unit_total
    
    print("\nRunning Integration Tests...")
    integ = IntegrationTestSuite()
    integ.run_all_tests()
    
    integ_passed = sum(1 for r in integ.test_results if r["passed"])
    integ_total = len(integ.test_results)
    total_passed += integ_passed
    total_count += integ_total
    
    print("\nRunning E2E Tests...")
    e2e = DataMenderE2ETest()
    e2e.run_all_tests()
    
    e2e_passed = sum(1 for r in e2e.test_results if r["passed"])
    e2e_total = len(e2e.test_results)
    total_passed += e2e_passed
    total_count += e2e_total
    
    print("\n" + "="*40)
    print(f"GRAND TOTAL: {total_passed}/{total_count} tests passed")
    print("="*40)
    
    if total_passed < total_count:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
