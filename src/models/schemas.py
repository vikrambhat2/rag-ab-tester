from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List


class QueryScore(BaseModel):
    query: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float

    def overall(self) -> float:
        return (
            self.faithfulness
            + self.answer_relevance
            + self.context_precision
            + self.context_recall
        ) / 4


class VariantResult(BaseModel):
    variant_name: str
    scores: List[QueryScore]

    def avg(self, metric: str) -> float:
        return sum(getattr(s, metric) for s in self.scores) / len(self.scores)


class MetricComparison(BaseModel):
    metric: str
    control_avg: float
    challenger_avg: float
    delta: float
    p_value: float
    cohens_d: float
    ci_low: float
    ci_high: float
    significant: bool   # p < 0.05
    meaningful: bool    # |d| >= 0.2
    winner: str         # "control" | "challenger" | "no difference"


class ExperimentResult(BaseModel):
    experiment_name: str
    control_name: str
    challenger_name: str
    control: VariantResult
    challenger: VariantResult
    comparisons: List[MetricComparison]
    overall_winner: str


class ABReport(BaseModel):
    experiments: List[ExperimentResult] = Field(default_factory=list)
