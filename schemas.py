
from pydantic import BaseModel, Field
from typing import List

class ExtractedLegalTest(BaseModel):
    """A single legal test extracted from a case."""
    test_novelty: str = Field(description="Is the test new, a refinement, or an application of existing law?")
    extracted_test_summary: str = Field(description="A concise summary of the legal test.")
    source_paragraphs: str = Field(description="The exact paragraphs from the source text that contain the legal test.")

class LegalTestComparison(BaseModel):
    """A comparison of two legal tests."""
    more_rule_like_test: str = Field(description="Either 'Test A' or 'Test B' - whichever test is more rule-like.")
    reasoning: str = Field(description="The reasoning for the comparison, referring to tests as 'Test A' and 'Test B'.")
