#!/bin/bash

pytest -s -vvv --cov=src --cov-report=xml:tests/test_coverage_result.xml --junitxml=tests/test_result.xml tests/sample.py
rm -rf .pytest_cache .coverage
