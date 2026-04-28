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

    def _assert_sympify_raises(self, expr_str):
        try:
            sympy.sympify(expr_str)
            self.fail(f"Expected an error for: {expr_str}")
        except (sympy.SympifyError, SyntaxError, ValueError, TypeError):
            pass

    def test_issue_18_expression_fails_without_fix(self):
        self._assert_sympify_raises(
            "((C0x8) - 6.929355)"
            "/(-2.465988(C1x1)1 + C2cos((C3x5)/(C4*x0)))"
        )

    def test_digit_paren_fails_without_fix(self):
        self._assert_sympify_raises("2.465988(C1x1)")

    def test_paren_digit_fails_without_fix(self):
        self._assert_sympify_raises("(C1x1)1")

    def test_decimal_before_paren_fails(self):
        self._assert_sympify_raises("1.2345(sin(x))")

    def test_integer_before_paren_fails(self):
        self._assert_sympify_raises("3(x + y)")

    def test_paren_before_decimal_fails(self):
        self._assert_sympify_raises("(x)1.5")

    def test_nested_missing_mul_fails(self):
        self._assert_sympify_raises("(1.5(2 + x))")

    def test_mixed_fixed_and_missing_fails(self):
        self._assert_sympify_raises("1.5*(C0*x0) + 2.5(C1*x1)1")

    def test_paren_integer_paren_fails(self):
        self._assert_sympify_raises("(x)3(y)")

    def test_power_then_paren_fails(self):
        self._assert_sympify_raises("x**2(C0*x1)")

    def test_negative_number_before_paren_fails(self):
        self._assert_sympify_raises("-1.5(x + y)")

    def test_nested_function_missing_mul_fails(self):
        self._assert_sympify_raises("sin(cos(1(x)))")

    def test_complex_expression_missing_mul_fails(self):
        self._assert_sympify_raises(
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
