"""Exercise the real optional CEC dependency at its first logging milestone."""
import csv
from pathlib import Path

import numpy as np
import os
import tempfile
import unittest

from cpso_serial.benchmarks.cec2013_lsgo import CEC2013LSGOAdapter


class NativeLoggingTest(unittest.TestCase):
    def test_native_logger_boundary(self):
        try:
            from cec2013lsgo import cec2013 as native
        except ImportError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                self.check_boundary(native)
            finally:
                os.chdir(previous)

    def check_boundary(self, native):
        benchmark = native.Benchmark()
        fitness = benchmark.get_function(1)
        x = np.zeros(1000)
        y = np.ones(1000)
        expected = [fitness(x), fitness(y)]
        for _ in range(119998):
            fitness(x)
        path = Path("results_f1.csv")
        with path.open() as stream:
            fe, function, value = next(csv.reader(stream))
        assert (int(fe), int(function)) == (120000, 1)
        self.assertTrue(np.isclose(float(value), min(expected), rtol=1e-6))
        del benchmark
        original = path.read_bytes()

        problem = CEC2013LSGOAdapter().get_problem(1)
        for i in range(120002):
            assert problem.objective((x, y)[i % 2]) == expected[i % 2]
        assert path.read_bytes() == original
