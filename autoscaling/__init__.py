"""
SageMaker Endpoint Auto Scaling Test Package

This package provides tools for testing auto scaling behavior of SageMaker endpoints.
"""

from .test_autoscaling import AutoScalingTester

__version__ = "1.0.0"
__all__ = ["AutoScalingTester"]