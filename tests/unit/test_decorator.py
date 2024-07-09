import guidance
import re


class TestDedent:

    def test_string(self):
        @guidance(stateless=True, dedent=True)
        def string(lm):
            lm += """\
            first line
            second line
            third line"""
            return lm

        lm = guidance.models.Mock()
        lm += string()
        assert str(lm) == "first line\nsecond line\nthird line"

    def test_fstring(self):
        @guidance(stateless=True, dedent=True)
        def f_string(lm):
            lm += f"""\
            {{
                "name": "{guidance.gen('name', stop='"', max_tokens=1)}",
            }}
            """
            return lm

        lm = guidance.models.Mock()
        lm += f_string()
        assert re.fullmatch(r"\{\n    \"name\": \"[^\"]+\",\n\}\n", str(lm))
