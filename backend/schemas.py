from typing import Any

from pydantic import BaseModel, Field, model_validator


class GraphNodeInput(BaseModel):
    id: str
    features: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdgeInput(BaseModel):
    source: str
    target: str
    weight: float = 1.0


class GraphDataInput(BaseModel):
    nodes: list[GraphNodeInput] = Field(default_factory=list)
    edges: list[GraphEdgeInput] = Field(default_factory=list)
    root_id: str | None = None


class AnalyzeRequest(BaseModel):
    text: str | None = Field(default=None, description="Claim or article text to analyze.")
    graph_data: GraphDataInput | None = None
    k: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def ensure_payload_present(self) -> "AnalyzeRequest":
        if not (self.text and self.text.strip()) and self.graph_data is None:
            raise ValueError("Provide at least one of 'text' or 'graph_data'.")
        return self


class AnalyzeResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction: str
    confidence: float
    causal_factors: list[str]
    intervention: str
    graph_fake_probability: float
    intervention_nodes: list[str]
    score_history: list[float]
    reduction_pct: float
    model_status: str
    graph_summary: dict[str, Any]
