"""Pydantic schemas for the Mushroom Toxicity API."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_POISONOUS = {
    "cap_shape": "x",
    "cap_surface": "s",
    "cap_color": "n",
    "bruises": "t",
    "odor": "p",
    "gill_attachment": "f",
    "gill_spacing": "c",
    "gill_size": "n",
    "gill_color": "k",
    "stalk_shape": "e",
    "stalk_root": "e",
    "stalk_surface_above_ring": "s",
    "stalk_surface_below_ring": "s",
    "stalk_color_above_ring": "w",
    "stalk_color_below_ring": "w",
    "veil_color": "w",
    "ring_number": "o",
    "ring_type": "p",
    "spore_print_color": "k",
    "population": "s",
    "habitat": "u",
}

EXAMPLE_EDIBLE = {
    "cap_shape": "x",
    "cap_surface": "s",
    "cap_color": "y",
    "bruises": "t",
    "odor": "a",
    "gill_attachment": "f",
    "gill_spacing": "c",
    "gill_size": "b",
    "gill_color": "k",
    "stalk_shape": "e",
    "stalk_root": "c",
    "stalk_surface_above_ring": "s",
    "stalk_surface_below_ring": "s",
    "stalk_color_above_ring": "w",
    "stalk_color_below_ring": "w",
    "veil_color": "w",
    "ring_number": "o",
    "ring_type": "p",
    "spore_print_color": "n",
    "population": "n",
    "habitat": "g",
}


class MushroomFeatures(BaseModel):
    """The 21 active features used to classify a mushroom specimen."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        json_schema_extra={"examples": [EXAMPLE_POISONOUS, EXAMPLE_EDIBLE]},
    )

    cap_shape: str = Field(..., alias="cap-shape", min_length=1, max_length=1, description="b=bell, c=conical, x=convex, f=flat, k=knobbed, s=sunken")
    cap_surface: str = Field(..., alias="cap-surface", min_length=1, max_length=1, description="f=fibrous, g=grooves, y=scaly, s=smooth")
    cap_color: str = Field(..., alias="cap-color", min_length=1, max_length=1, description="n=brown, b=buff, c=cinnamon, g=gray, r=green, p=pink, u=purple, e=red, w=white, y=yellow")
    bruises: str = Field(..., min_length=1, max_length=1, description="t=bruises, f=no")
    odor: str = Field(..., min_length=1, max_length=1, description="a=almond, l=anise, c=creosote, y=fishy, f=foul, m=musty, n=none, p=pungent, s=spicy")
    gill_attachment: str = Field(..., alias="gill-attachment", min_length=1, max_length=1, description="a=attached, d=descending, f=free, n=notched")
    gill_spacing: str = Field(..., alias="gill-spacing", min_length=1, max_length=1, description="c=close, w=crowded, d=distant")
    gill_size: str = Field(..., alias="gill-size", min_length=1, max_length=1, description="b=broad, n=narrow")
    gill_color: str = Field(..., alias="gill-color", min_length=1, max_length=1, description="k=black, n=brown, b=buff, h=chocolate, g=gray, r=green, o=orange, p=pink, u=purple, e=red, w=white, y=yellow")
    stalk_shape: str = Field(..., alias="stalk-shape", min_length=1, max_length=1, description="e=enlarging, t=tapering")
    stalk_root: str = Field(..., alias="stalk-root", min_length=1, max_length=1, description="b=bulbous, c=club, u=cup, e=equal, z=rhizomorphs, r=rooted, ?=missing")
    stalk_surface_above_ring: str = Field(..., alias="stalk-surface-above-ring", min_length=1, max_length=1, description="f=fibrous, y=scaly, k=silky, s=smooth")
    stalk_surface_below_ring: str = Field(..., alias="stalk-surface-below-ring", min_length=1, max_length=1, description="f=fibrous, y=scaly, k=silky, s=smooth")
    stalk_color_above_ring: str = Field(..., alias="stalk-color-above-ring", min_length=1, max_length=1, description="n=brown, b=buff, c=cinnamon, g=gray, o=orange, p=pink, e=red, w=white, y=yellow")
    stalk_color_below_ring: str = Field(..., alias="stalk-color-below-ring", min_length=1, max_length=1, description="n=brown, b=buff, c=cinnamon, g=gray, o=orange, p=pink, e=red, w=white, y=yellow")
    veil_color: str = Field(..., alias="veil-color", min_length=1, max_length=1, description="n=brown, o=orange, w=white, y=yellow")
    ring_number: str = Field(..., alias="ring-number", min_length=1, max_length=1, description="n=none, o=one, t=two")
    ring_type: str = Field(..., alias="ring-type", min_length=1, max_length=1, description="c=cobwebby, e=evanescent, f=flaring, l=large, n=none, p=pendant, s=sheathing, z=zone")
    spore_print_color: str = Field(..., alias="spore-print-color", min_length=1, max_length=1, description="k=black, n=brown, b=buff, h=chocolate, r=green, o=orange, u=purple, w=white, y=yellow")
    population: str = Field(..., min_length=1, max_length=1, description="a=abundant, c=clustered, n=numerous, s=scattered, v=several, y=solitary")
    habitat: str = Field(..., min_length=1, max_length=1, description="g=grasses, l=leaves, m=meadows, p=paths, u=urban, w=waste, d=woods")


class MushroomPrediction(BaseModel):
    is_poisonous: bool = Field(description="True if poison_probability >= threshold")
    poison_probability: float = Field(description="Predicted probability of being poisonous (0.0 to 1.0)")
    risk: Literal["low", "medium", "high"] = Field(description="Safety risk assessment")
    threshold: float = Field(description="Safety threshold tuned during cross-validation")
    recommendation: str = Field(description="Actionable advice for foragers")


class ModelInfo(BaseModel):
    name: str
    loaded: bool
    algorithm: str | None = None
    trained_at: str | None = None
    metrics: dict[str, float] = {}
