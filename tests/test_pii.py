"""
Cognitive Core — P-003: PII Redaction Tests

Tests:
  - test_ssn_redacted — SSN patterns masked
  - test_account_number_redacted — 16-digit numbers masked
  - test_name_from_case_redacted — names from case input masked
  - test_email_redacted — email addresses masked
  - test_phone_redacted — phone numbers masked
  - test_round_trip_consistent — redact → deredact produces original
  - test_redaction_map_logged — audit summary has counts, no PII
  - test_redaction_disabled_per_domain — disabled domain skips redaction
  - test_nested_pii — PII inside JSON structures caught
  - test_multiple_occurrences — same PII value gets same placeholder
  - test_partial_name_match — first and last names caught separately
  - test_credit_card_redacted — card numbers masked
  - test_dob_redacted — dates of birth masked
  - test_no_false_positives — regular numbers not masked
  - test_empty_input — empty string passes through
  - test_deredact_handles_rearrangement — LLM reorders placeholders
"""

import os
import sys
import unittest

# Import PII module directly
import importlib.util
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pii_path = os.path.join(_base, "engine", "pii.py")
_spec = importlib.util.spec_from_file_location("engine.pii", _pii_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.pii"] = _mod
_spec.loader.exec_module(_mod)

PiiRedactor = _mod.PiiRedactor
RedactionMap = _mod.RedactionMap
create_redactor_from_domain = _mod.create_redactor_from_domain


class TestSsnRedacted(unittest.TestCase):
    """test_ssn_redacted — '123-45-6789' → '[SSN_1]' in prompt, restored in output"""

    def test_ssn_masked(self):
        r = PiiRedactor()
        text = "Member SSN is 123-45-6789 and secondary is 987-65-4321."
        redacted = r.redact(text)

        self.assertNotIn("123-45-6789", redacted)
        self.assertNotIn("987-65-4321", redacted)
        self.assertIn("[SSN_1]", redacted)
        self.assertIn("[SSN_2]", redacted)

    def test_ssn_round_trip(self):
        r = PiiRedactor()
        original = "SSN: 123-45-6789"
        redacted = r.redact(original)
        restored = r.deredact(redacted)
        self.assertEqual(restored, original)


class TestAccountNumberRedacted(unittest.TestCase):
    """test_account_number_redacted — 10-16 digit numbers masked"""

    def test_16_digit_masked(self):
        r = PiiRedactor()
        text = "Account 1234567890123456 has a balance."
        redacted = r.redact(text)
        self.assertNotIn("1234567890123456", redacted)
        # 16-digit number matches credit card pattern (higher priority)
        self.assertTrue("[CC_" in redacted or "[ACCT_" in redacted)

    def test_10_digit_masked(self):
        r = PiiRedactor()
        text = "Member ID: 1234567890."
        redacted = r.redact(text)
        self.assertNotIn("1234567890", redacted)


class TestNameFromCaseRedacted(unittest.TestCase):
    """test_name_from_case_redacted — 'Emily Torres' from case input → '[NAME_1]'"""

    def test_name_masked(self):
        r = PiiRedactor()
        r.register_entities_from_case({
            "get_customer": {"name": "Emily Torres", "member_id": "M12345"}
        })
        text = "Customer Emily Torres filed a return request."
        redacted = r.redact(text)

        self.assertNotIn("Emily Torres", redacted)
        self.assertIn("[NAME_", redacted)

    def test_name_round_trip(self):
        r = PiiRedactor()
        r.register_entities_from_case({"get_customer": {"name": "John Smith"}})
        original = "Dear John Smith, your return is approved."
        redacted = r.redact(original)
        restored = r.deredact(redacted)
        self.assertEqual(restored, original)


class TestEmailRedacted(unittest.TestCase):
    """test_email_redacted — email addresses masked"""

    def test_email_masked(self):
        r = PiiRedactor()
        text = "Contact: user@example.com or admin@company.org"
        redacted = r.redact(text)
        self.assertNotIn("user@example.com", redacted)
        self.assertNotIn("admin@company.org", redacted)
        self.assertIn("[EMAIL_1]", redacted)
        self.assertIn("[EMAIL_2]", redacted)

    def test_email_round_trip(self):
        r = PiiRedactor()
        original = "Email: test@navy.mil"
        redacted = r.redact(original)
        restored = r.deredact(redacted)
        self.assertEqual(restored, original)


class TestPhoneRedacted(unittest.TestCase):
    """test_phone_redacted — phone numbers masked"""

    def test_phone_masked(self):
        r = PiiRedactor()
        text = "Call 555-123-4567 or (800) 555-0199"
        redacted = r.redact(text)
        self.assertNotIn("555-123-4567", redacted)
        self.assertNotIn("(800) 555-0199", redacted)

    def test_phone_round_trip(self):
        r = PiiRedactor()
        original = "Phone: 703-555-1234"
        redacted = r.redact(original)
        restored = r.deredact(redacted)
        self.assertEqual(restored, original)


class TestRoundTripConsistent(unittest.TestCase):
    """test_round_trip_consistent — redact → LLM → de-redact produces correct output"""

    def test_complex_round_trip(self):
        r = PiiRedactor()
        r.register_entities_from_case({
            "get_customer": {
                "name": "Maria Rodriguez",
                "address": "456 Oak Street, Springfield IL 62704",
            },
            "get_order": {"order_id": "ORD-12345"},
        })

        prompt = (
            "Maria Rodriguez at 456 Oak Street, Springfield IL 62704 "
            "ordered item ORD-12345. Her SSN is 111-22-3333 and email is "
            "maria@test.com. Phone: 555-888-9999."
        )
        redacted = r.redact(prompt)

        # Verify nothing PII in redacted
        self.assertNotIn("Maria Rodriguez", redacted)
        self.assertNotIn("456 Oak Street", redacted)
        self.assertNotIn("111-22-3333", redacted)
        self.assertNotIn("maria@test.com", redacted)
        self.assertNotIn("555-888-9999", redacted)

        # Simulate LLM response using the same placeholders
        llm_response = f"Approved return for {r._map._forward.get('Maria Rodriguez', '[?]')}"
        restored = r.deredact(llm_response)
        self.assertIn("Maria Rodriguez", restored)


class TestRedactionMapLogged(unittest.TestCase):
    """test_redaction_map_logged — audit entry shows field count, no actual PII"""

    def test_audit_summary_no_pii(self):
        r = PiiRedactor()
        r.register_entities_from_case({"get_customer": {"name": "Bob Lee"}})
        r.redact("Bob Lee with SSN 111-22-3333 and email bob@test.com")

        summary = r.audit_summary
        self.assertGreater(summary["total_redacted"], 0)
        self.assertIn("SSN", summary["by_type"])
        self.assertIn("EMAIL", summary["by_type"])

        # Summary must NOT contain actual PII
        summary_str = str(summary)
        self.assertNotIn("Bob Lee", summary_str)
        self.assertNotIn("111-22-3333", summary_str)
        self.assertNotIn("bob@test.com", summary_str)


class TestRedactionDisabledPerDomain(unittest.TestCase):
    """test_redaction_disabled_per_domain — domain with pii_redaction: false skips"""

    def test_disabled_passes_through(self):
        r = PiiRedactor(enabled=False)
        text = "SSN: 123-45-6789 Name: John Smith"
        redacted = r.redact(text)
        self.assertEqual(redacted, text)  # No change

    def test_disabled_deredact_passes_through(self):
        r = PiiRedactor(enabled=False)
        text = "[SSN_1] placeholder"
        self.assertEqual(r.deredact(text), text)

    def test_domain_config_disables(self):
        r = create_redactor_from_domain({"pii_redaction": False})
        self.assertFalse(r.enabled)
        text = "SSN: 123-45-6789"
        self.assertEqual(r.redact(text), text)

    def test_domain_config_default_enabled(self):
        r = create_redactor_from_domain({})
        self.assertTrue(r.enabled)


class TestNestedPii(unittest.TestCase):
    """test_nested_pii — PII inside JSON structures within the prompt caught"""

    def test_pii_in_json_string(self):
        r = PiiRedactor()
        r.register_entities_from_case({
            "get_customer": {"name": "Alice Johnson", "member_id": "M99887766"}
        })
        text = '{"customer": {"name": "Alice Johnson", "ssn": "555-66-7777"}}'
        redacted = r.redact(text)
        self.assertNotIn("Alice Johnson", redacted)
        self.assertNotIn("555-66-7777", redacted)

    def test_deeply_nested_entities(self):
        r = PiiRedactor()
        r.register_entities_from_case({
            "get_order": {
                "shipping": {
                    "address": {
                        "street": "123 Main St",
                        "city": "Anytown",
                    }
                }
            }
        })
        text = "Ship to 123 Main St, Anytown"
        redacted = r.redact(text)
        self.assertNotIn("123 Main St", redacted)


class TestMultipleOccurrences(unittest.TestCase):
    """Same PII value gets the same placeholder everywhere."""

    def test_same_ssn_same_placeholder(self):
        r = PiiRedactor()
        text = "SSN 123-45-6789 confirmed. Repeat: 123-45-6789."
        redacted = r.redact(text)

        # Should have exactly one SSN placeholder used twice
        self.assertEqual(redacted.count("[SSN_1]"), 2)

    def test_same_name_same_placeholder(self):
        r = PiiRedactor()
        r.register_entities_from_case({"get_customer": {"name": "Bob Lee"}})
        text = "Bob Lee said Bob Lee wants a refund."
        redacted = r.redact(text)
        # Full name placeholder used twice
        self.assertNotIn("Bob Lee", redacted)


class TestPartialNameMatch(unittest.TestCase):
    """First and last names caught separately."""

    def test_partial_name_caught(self):
        r = PiiRedactor()
        r.register_entities_from_case({"get_customer": {"name": "Emily Torres"}})
        text = "Emily filed a claim. Torres confirmed."
        redacted = r.redact(text)
        self.assertNotIn("Emily", redacted)
        self.assertNotIn("Torres", redacted)


class TestCreditCardRedacted(unittest.TestCase):
    """Credit card numbers masked."""

    def test_cc_masked(self):
        r = PiiRedactor()
        text = "Card ending 4111-1111-1111-1111 was charged."
        redacted = r.redact(text)
        self.assertNotIn("4111-1111-1111-1111", redacted)
        self.assertIn("[CC_", redacted)


class TestDobRedacted(unittest.TestCase):
    """Dates of birth masked."""

    def test_dob_masked(self):
        r = PiiRedactor()
        text = "DOB: 03/15/1990"
        redacted = r.redact(text)
        self.assertNotIn("03/15/1990", redacted)
        self.assertIn("[DOB_", redacted)


class TestNoFalsePositives(unittest.TestCase):
    """Regular numbers and short words not falsely masked."""

    def test_short_numbers_not_masked(self):
        r = PiiRedactor()
        text = "Order has 5 items totaling $299.99 with 30-day window."
        redacted = r.redact(text)
        self.assertEqual(redacted, text)  # No PII to redact

    def test_case_id_not_masked(self):
        r = PiiRedactor()
        text = "Case N01 processed in step 3."
        redacted = r.redact(text)
        self.assertEqual(redacted, text)


class TestEmptyInput(unittest.TestCase):
    """Empty strings pass through."""

    def test_empty_string(self):
        r = PiiRedactor()
        self.assertEqual(r.redact(""), "")
        self.assertEqual(r.deredact(""), "")


class TestDeredactHandlesRearrangement(unittest.TestCase):
    """LLM may reorder placeholders in its response."""

    def test_rearranged_placeholders(self):
        r = PiiRedactor()
        r.register_entities_from_case({"get_customer": {"name": "Jane Doe"}})
        prompt = "Jane Doe has SSN 222-33-4444 and email jane@test.com"
        redacted = r.redact(prompt)

        # LLM responds with placeholders in different order
        name_ph = r._map._forward["Jane Doe"]
        ssn_ph = r._map._forward["222-33-4444"]
        email_ph = r._map._forward["jane@test.com"]

        llm_response = f"Email {email_ph} belongs to {name_ph} (SSN: {ssn_ph})"
        restored = r.deredact(llm_response)

        self.assertIn("jane@test.com", restored)
        self.assertIn("Jane Doe", restored)
        self.assertIn("222-33-4444", restored)


class TestRedactorCount(unittest.TestCase):
    """Redaction count tracks unique values."""

    def test_count_tracks_unique(self):
        r = PiiRedactor()
        r.redact("SSN 111-22-3333 and 111-22-3333 and 444-55-6666")
        self.assertEqual(r.redaction_count, 2)  # 2 unique SSNs


if __name__ == "__main__":
    unittest.main()


class TestPiiIntegrationInNodePath(unittest.TestCase):
    """Verify PII redaction is wired into the node execution path."""

    def test_pii_module_importable_from_nodes(self):
        """nodes.py can import PiiRedactor."""
        try:
            from engine.pii import PiiRedactor
            r = PiiRedactor()
            self.assertTrue(r.enabled)
        except ImportError:
            self.fail("PiiRedactor not importable")

    def test_redact_then_deredact_in_prompt_flow(self):
        """Simulate the create_node flow: case input → redact prompt → LLM → de-redact response."""
        r = PiiRedactor()

        # Step 1: Register entities from case input (like node_fn does)
        case_input = {
            "get_member": {"name": "John Smith", "member_id": "M12345"},
            "get_transactions": {"account_id": "9876543210"},
            "complaint": "My card 4111-1111-1111-1111 was charged."
        }
        r.register_entities_from_case(case_input)

        # Step 2: Build prompt (like render_prompt does)
        prompt = (
            "Customer John Smith (member M12345) filed a complaint about "
            "card 4111-1111-1111-1111 on account 9876543210. "
            "SSN on file: 123-45-6789. Email: john@test.com."
        )

        # Step 3: Redact (what nodes.py now does before LLM call)
        redacted = r.redact(prompt)

        # Verify PII is gone from what the LLM sees
        self.assertNotIn("John Smith", redacted)
        self.assertNotIn("4111-1111-1111-1111", redacted)
        self.assertNotIn("123-45-6789", redacted)
        self.assertNotIn("john@test.com", redacted)
        self.assertNotIn("9876543210", redacted)

        # Step 4: Simulate LLM response using placeholders
        name_ph = r._map._forward.get("John Smith", "[NAME_?]")
        llm_response = f'{{"reasoning": "Approved refund for {name_ph}", "confidence": 0.95}}'

        # Step 5: De-redact (what nodes.py now does after LLM response)
        restored = r.deredact(llm_response)

        self.assertIn("John Smith", restored)
        self.assertNotIn("[NAME_", restored)
