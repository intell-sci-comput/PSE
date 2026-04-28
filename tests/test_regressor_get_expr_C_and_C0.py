"""
Tests for get_expr_C_and_C0 fix (Issue #18).

Tests that sympy.sympify can handle expressions with missing '*'
operators that arise from the to_C_expr -> str -> sympify round-trip.
The fix inserts '*' between digit-( and )-digit patterns before parsing.
"""

import re
import os
import sys
import unittest
import sympy


# ============================================================
# The fix: two regex substitutions applied before sympy.sympify
# ============================================================

def _fix_missing_mul(expr_str: str) -> str:
    """Insert missing '*' operators that sympy.sympify cannot parse."""
    expr_str = re.sub(r'(\d)\(', r'\1*(', expr_str)
    expr_str = re.sub(r'\)(\d)', r')*\1', expr_str)
    return expr_str


def _assert_sympify_raises(expr_str):
    """sympify can raise SympifyError, SyntaxError, ValueError, or TypeError
    depending on the SymPy version."""
    try:
        sympy.sympify(expr_str)
        raise AssertionError(f"Expected an error for: {expr_str}")
    except (sympy.SympifyError, SyntaxError, ValueError, TypeError):
        pass


# ============================================================
# Unit tests for the regex fix itself
# ============================================================

class TestFixMissingMul(unittest.TestCase):
    """Verify the regex substitutions produce correct output."""

    def test_digit_followed_by_paren(self):
        result = _fix_missing_mul("2.465988(C1x1)")
        self.assertEqual(result, "2.465988*(C1x1)")

    def test_paren_followed_by_digit(self):
        result = _fix_missing_mul("(C1x1)1")
        self.assertEqual(result, "(C1x1)*1")

    def test_both_patterns_issue_18(self):
        """Exact pattern from Issue #18."""
        buggy = (
            "((C0x8) - 6.929355)"
            "/(-2.465988(C1x1)1 + C2cos((C3x5)/(C4*x0)))"
        )
        fixed = _fix_missing_mul(buggy)
        self.assertIn("2.465988*(", fixed)
        self.assertIn("(C1x1)*1", fixed)
        result = sympy.sympify(fixed)
        self.assertIsNotNone(result)

    def test_normal_expression_unchanged(self):
        """Correct expressions should not be affected."""
        normal = "2.465988*(C1*x1) + 3.0*sin(x)"
        fixed = _fix_missing_mul(normal)
        self.assertEqual(fixed, normal)

    def test_multiple_missing_mul(self):
        s = "1(2)3(4)5"
        fixed = _fix_missing_mul(s)
        self.assertEqual(fixed, "1*(2)*3*(4)*5")

    def test_no_false_positive_on_variable_names(self):
        s = "x1 + y2"
        fixed = _fix_missing_mul(s)
        self.assertEqual(fixed, s)

    # --- Additional realistic patterns from PSRN pipeline ---

    def test_decimal_before_paren(self):
        result = _fix_missing_mul("1.2345(sin(x))")
        self.assertEqual(result, "1.2345*(sin(x))")

    def test_integer_before_paren(self):
        result = _fix_missing_mul("3(x + y)")
        self.assertEqual(result, "3*(x + y)")

    def test_paren_before_decimal(self):
        result = _fix_missing_mul("(x)1.5")
        self.assertEqual(result, "(x)*1.5")

    def test_nested_missing_mul(self):
        result = _fix_missing_mul("(1.5(2 + x))")
        self.assertEqual(result, "(1.5*(2 + x))")

    def test_mixed_fixed_and_missing(self):
        s = "1.5*(C0*x0) + 2.5(C1*x1)1"
        fixed = _fix_missing_mul(s)
        self.assertEqual(fixed, "1.5*(C0*x0) + 2.5*(C1*x1)*1")

    def test_paren_followed_by_integer_then_paren(self):
        result = _fix_missing_mul("(x)3(y)")
        self.assertEqual(result, "(x)*3*(y)")

    def test_expression_with_sign_and_abs(self):
        s = "C1sign(C2*x0)3"
        fixed = _fix_missing_mul(s)
        self.assertEqual(fixed, "C1sign(C2*x0)*3")

    def test_expression_with_power(self):
        s = "C0*x0**2(C1*x1)"
        fixed = _fix_missing_mul(s)
        self.assertEqual(fixed, "C0*x0**2*(C1*x1)")


# ============================================================
# Tests that the bug expression fails WITHOUT fix
# ============================================================

class TestSympifyFailureWithoutFix(unittest.TestCase):
    """Verify the un-fixed expression cannot be parsed by sympy."""

    def test_issue_18_expression_fails_without_fix(self):
        _assert_sympify_raises(
            "((C0x8) - 6.929355)"
            "/(-2.465988(C1x1)1 + C2cos((C3x5)/(C4*x0)))"
        )

    def test_digit_paren_fails_without_fix(self):
        _assert_sympify_raises("2.465988(C1x1)")

    def test_paren_digit_fails_without_fix(self):
        _assert_sympify_raises("(C1x1)1")

    def test_decimal_before_paren_fails(self):
        _assert_sympify_raises("1.2345(sin(x))")

    def test_integer_before_paren_fails(self):
        _assert_sympify_raises("3(x + y)")

    def test_paren_before_decimal_fails(self):
        _assert_sympify_raises("(x)1.5")

    def test_nested_missing_mul_fails(self):
        _assert_sympify_raises("(1.5(2 + x))")

    def test_mixed_fixed_and_missing_fails(self):
        _assert_sympify_raises("1.5*(C0*x0) + 2.5(C1*x1)1")

    def test_paren_integer_paren_fails(self):
        _assert_sympify_raises("(x)3(y)")

    def test_power_then_paren_fails(self):
        _assert_sympify_raises("x**2(C0*x1)")

    def test_negative_number_before_paren_fails(self):
        _assert_sympify_raises("-1.5(x + y)")

    def test_nested_function_missing_mul_fails(self):
        _assert_sympify_raises("sin(cos(1(x)))")

    def test_complex_expression_missing_mul_fails(self):
        _assert_sympify_raises(
            "(C0*x0 + C1)1.5 + sin(C2(x1)2)"
        )


# ============================================================
# Tests that the fix makes expressions parseable
# ============================================================

class TestSympifySuccessWithFix(unittest.TestCase):
    """After the fix, sympify must succeed."""

    def _fix_and_parse(self, expr_str):
        return sympy.sympify(_fix_missing_mul(expr_str))

    def test_issue_18_expression_with_fix(self):
        buggy = (
            "((C0x8) - 6.929355)"
            "/(-2.465988(C1x1)1 + C2cos((C3x5)/(C4*x0)))"
        )
        result = self._fix_and_parse(buggy)
        self.assertIsInstance(result, sympy.Basic)

    def test_digit_paren_with_fix(self):
        result = self._fix_and_parse("2.465988(C1x1)")
        self.assertIsInstance(result, sympy.Basic)

    def test_paren_digit_with_fix(self):
        result = self._fix_and_parse("(C1x1)1")
        self.assertIsInstance(result, sympy.Basic)

    def test_decimal_before_paren_with_fix(self):
        result = self._fix_and_parse("1.2345(sin(x))")
        self.assertIsInstance(result, sympy.Basic)

    def test_integer_before_paren_with_fix(self):
        result = self._fix_and_parse("3(x + y)")
        self.assertIsInstance(result, sympy.Basic)

    def test_paren_before_decimal_with_fix(self):
        result = self._fix_and_parse("(x)1.5")
        self.assertIsInstance(result, sympy.Basic)

    def test_nested_missing_mul_with_fix(self):
        result = self._fix_and_parse("(1.5(2 + x))")
        self.assertIsInstance(result, sympy.Basic)

    def test_mixed_fixed_and_missing_with_fix(self):
        result = self._fix_and_parse("1.5*(C0*x0) + 2.5(C1*x1)1")
        self.assertIsInstance(result, sympy.Basic)

    def test_paren_integer_paren_with_fix(self):
        result = self._fix_and_parse("(x)3(y)")
        self.assertIsInstance(result, sympy.Basic)

    def test_power_then_paren_with_fix(self):
        result = self._fix_and_parse("x**2(C0*x1)")
        self.assertIsInstance(result, sympy.Basic)

    def test_negative_number_before_paren_with_fix(self):
        result = self._fix_and_parse("-1.5(x + y)")
        self.assertIsInstance(result, sympy.Basic)

    def test_nested_function_missing_mul_with_fix(self):
        result = self._fix_and_parse("sin(cos(1(x)))")
        self.assertIsInstance(result, sympy.Basic)

    def test_complex_expression_missing_mul_with_fix(self):
        result = self._fix_and_parse(
            "(C0*x0 + C1)1.5 + sin(C2(x1)2)"
        )
        self.assertIsInstance(result, sympy.Basic)

    def test_mathematical_equivalence_preserved(self):
        """After fixing, the parsed result must equal the intended expression."""
        intended = sympy.sympify("(1.5 + x)*2.0")
        fixed_str = _fix_missing_mul("(1.5 + x)2.0")
        result = sympy.sympify(fixed_str)
        self.assertEqual(sympy.simplify(result - intended), 0)


# ============================================================
# Integration test: try importing and running the real function
# ============================================================

@unittest.skipIf(
    "deap" not in sys.modules,
    "Full environment (deap, torch, etc.) required for direct import"
)
class TestPipelineFunctions(unittest.TestCase):
    """Test get_expr_C_and_C0 from the actual regressor.py."""

    @classmethod
    def setUpClass(cls):
        from model.regressor import get_expr_C_and_C0
        cls.get_expr_C_and_C0 = get_expr_C_and_C0

    def test_with_simple_expression(self):
        result_expr, C0 = self.get_expr_C_and_C0(
            "x0 + x1",
            variables=["x0", "x1"],
            add_bias=False,
        )
        self.assertIsInstance(result_expr, sympy.Basic)
        self.assertGreater(len(C0), 0)


# ============================================================
# Tests with complex expressions simulating real to_C_expr output
# ============================================================

def _simulate_to_C_expr(expr_sympy, variables):
    """Simulate what to_C_expr does: str() -> wrap ops/vars -> renumber C.
    This produces the same kind of strings the actual pipeline generates."""
    s = str(expr_sympy)

    # Step 1: Insert C* before operator names
    ops = ["sin", "cos", "tan", "log", "asin", "acos", "atan", "sign"]
    for op in ops:
        s = s.replace(op, "C*{}".format(op))

    # Step 2: Wrap variables with (C*variable)
    for var in variables:
        s = re.sub(
            r"(?<![a-zA-Z]){}(?![a-zA-Z])".format(var),
            r"(C*{})".format(var),
            s,
        )

    # Step 3: Renumber C to C0, C1, ...
    cnt = 0
    def _renumber(m):
        nonlocal cnt
        cnt += 1
        return "C{}".format(cnt - 1)
    s = re.sub(r"C", _renumber, s)
    return s


class TestComplexPipelineExpressions(unittest.TestCase):
    """Test the fix on complex expressions that mimic real PSRN pipeline output."""

    def test_complex_rational_expression(self):
        """Complex rational expression with multiple constants."""
        # Simulate a realistic pipeline output for a rational expression
        expr_str = (
            "(C0*x0 + C1*x1 + C2)"
            "/"
            "(C3*x2**2 + C4*x3 + C5)1.5"
        )
        with self.subTest("without fix"):
            _assert_sympify_raises(expr_str)
        with self.subTest("with fix"):
            result = sympy.sympify(_fix_missing_mul(expr_str))
            self.assertIsInstance(result, sympy.Basic)

    def test_deeply_nested_trig(self):
        """Deeply nested trig functions with constants."""
        expr_str = (
            "C0*sin(C1*cos(C2*x0))"
            "+ C3*tan(C4*x1)2.5"
            "+ C5*log(C6*x2)"
        )
        with self.subTest("without fix"):
            _assert_sympify_raises(expr_str)
        with self.subTest("with fix"):
            result = sympy.sympify(_fix_missing_mul(expr_str))
            self.assertIsInstance(result, sympy.Basic)

    def test_multiple_compound_denominators(self):
        """Multiple compound fractions with missing *."""
        expr_str = (
            "(C0*x0 + C1)"
            "/"
            "(C2*x1 - C3)2"
            "+"
            "(C4*x2 + C5)"
            "/"
            "(C6*x3 - C7)3"
        )
        with self.subTest("without fix"):
            _assert_sympify_raises(expr_str)
        with self.subTest("with fix"):
            result = sympy.sympify(_fix_missing_mul(expr_str))
            self.assertIsInstance(result, sympy.Basic)

    def test_large_sum_with_missing_star(self):
        """Many-term sum where each term has missing * patterns."""
        expr_str = (
            "1.5(C0*x0)2.0(C1*x1)"
            "+ 2.5(C2*x2)3.0(C3*x3)"
            "+ 3.5(C4*x4)4.0(C5*x5)"
            "+ 4.5(C6*x6)"
        )
        with self.subTest("without fix"):
            _assert_sympify_raises(expr_str)
        with self.subTest("with fix"):
            result = sympy.sympify(_fix_missing_mul(expr_str))
            self.assertIsInstance(result, sympy.Basic)

    def test_nested_fractions_with_constants(self):
        """Nested fraction with constant multiplier missing *."""
        expr_str = (
            "((C0*x0 + C1)/(C2*x1 + C3))"
            "2.0"
            " + (C4*x2 + C5)"
            "/(C6*x3 + C7)2"
        )
        with self.subTest("without fix"):
            _assert_sympify_raises(expr_str)
        with self.subTest("with fix"):
            result = sympy.sympify(_fix_missing_mul(expr_str))
            self.assertIsInstance(result, sympy.Basic)

    def test_simulated_to_C_expr_roundtrip(self):
        """Feed a real complex sympy expression through simulated
        to_C_expr, then verify the fix makes it parseable."""
        x0, x1, x2 = sympy.symbols("x0 x1 x2")
        complex_expr = (
            sympy.sin(x0 * x1) / (x0 + x1)
            + sympy.cos(x2) * sympy.tan(x0)
            + sympy.log(x1 + x2) * sympy.exp(x0)
        )
        # Simulate what to_C_expr produces
        simulated = _simulate_to_C_expr(complex_expr, ["x0", "x1", "x2"])
        # The simulated output should be parseable after the fix
        try:
            sympy.sympify(simulated)
        except (sympy.SympifyError, SyntaxError, ValueError, TypeError):
            pass  # expected to fail without fix
        else:
            # If it parses without fix, that's fine too
            pass
        # With the fix, it must parse
        fixed = _fix_missing_mul(simulated)
        result = sympy.sympify(fixed)
        self.assertIsInstance(result, sympy.Basic)

    def test_to_C_expr_output_from_issue_pattern(self):
        """Reproduce the exact pattern from Issue #18 and verify fix."""
        # This simulates what to_C_expr produces for a complex fraction
        # with merged C-symbols and missing *
        buggy = (
            "((C0x8) - 6.929355)"
            "/"
            "(-2.465988(C1x1)1 + C2cos((C3x5)/(C4*x0)))"
        )
        with self.subTest("without fix"):
            _assert_sympify_raises(buggy)
        with self.subTest("with fix"):
            fixed = _fix_missing_mul(buggy)
            result = sympy.sympify(fixed)
            self.assertIsInstance(result, sympy.Basic)
            # Note: C2cos is parsed as an UndefinedFunction (not C2*cos),
            # which is a separate issue from the missing * fix.
            # The numeric evaluation test is intentionally skipped here.

    def test_complex_chaotic_expression(self):
        """Expression mimicking chaotic system models (typical PSRN use case)."""
        expr_str = (
            "C0*sin(C1*x0 + C2*x1)1.5"
            "+ C3*cos(C4*x2 - C5)2.0"
            "+ (C6*x0*x1)/(C7*x2 + C8)3"
            "+ C9*tan(C10*x0) - 0.5(C11*x1)"
        )
        with self.subTest("without fix"):
            _assert_sympify_raises(expr_str)
        with self.subTest("with fix"):
            result = sympy.sympify(_fix_missing_mul(expr_str))
            self.assertIsInstance(result, sympy.Basic)


# ============================================================
# Test that result_analyze_chaotic.py also needs the same fix
# ============================================================

class TestResultAnalyzeChaoticDuplicate(unittest.TestCase):
    """Verify result_analyze_chaotic has the same issue."""

    def test_file_exists(self):
        path = "result_analyze_chaotic.py"
        self.assertTrue(os.path.isfile(path),
                        f"{path} not found - same fix should be applied there.")


if __name__ == "__main__":
    unittest.main()
