# Pydantic model for structured restaurant data output
from pydantic import BaseModel

class RestaurantOutput(BaseModel):
    """Structured output model for restaurant name and menu items"""
    restaurant_name: str
    menu_items: str