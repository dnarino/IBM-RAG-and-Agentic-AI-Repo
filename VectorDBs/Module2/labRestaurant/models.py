import hashlib
from typing import List, Dict, Any
from pydantic import BaseModel, Field, model_validator

class FoodItem(BaseModel):
    food_id: str = ""
    food_name: str
    food_description: str = ""
    food_calories_per_serving: int = 0
    food_ingredients: List[str] = Field(default_factory=list)
    cuisine_type: str = "Unknown"
    cooking_method: str = ""
    food_health_benefits: str = ""
    food_nutritional_factors: Dict[str, str] = Field(default_factory=dict)
    food_features: Dict[str, Any] = Field(default_factory=dict)
    taste_profile: str = ""

    @model_validator(mode="before")
    @classmethod
    def preprocess_food_item(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # Extract taste profile from nested food_features if available
        features = data.get("food_features")
        if isinstance(features, dict):
            taste_vals = [str(v) for v in features.values() if v]
            data["taste_profile"] = ", ".join(taste_vals)
        else:
            data["taste_profile"] = ""

        return data

    def get_deterministic_id(self) -> str:
        """Generate a deterministic MD5 hash based on food_id, name, cuisine, and calories."""
        unique_string = f"{self.food_id}|{self.food_name.lower().strip()}|{self.cuisine_type.lower().strip()}|{self.food_calories_per_serving}"
        return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

    def to_embedding_text(self) -> str:
        """Construct comprehensive string representation for semantic embedding generation."""
        text = f"Name: {self.food_name}. "
        if self.food_description:
            text += f"Description: {self.food_description}. "
        if self.food_ingredients:
            text += f"Ingredients: {', '.join(self.food_ingredients)}. "
        text += f"Cuisine: {self.cuisine_type}. "
        if self.cooking_method:
            text += f"Cooking method: {self.cooking_method}. "
        if self.taste_profile:
            text += f"Taste and features: {self.taste_profile}. "
        if self.food_health_benefits:
            text += f"Health benefits: {self.food_health_benefits}. "
        if self.food_nutritional_factors:
            nutrition_text = ', '.join([f"{k}: {v}" for k, v in self.food_nutritional_factors.items()])
            text += f"Nutrition: {nutrition_text}."
        return text
