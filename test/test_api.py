from unittest.mock import patch

import numpy as np
import pytest
from httpx import AsyncClient

from starter.main import app


@pytest.mark.asyncio
async def test_say_hello():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting": "Hello World!"}

@pytest.mark.asyncio
@patch('starter.main.make_prediction')
async def test_create_item(mock_make_prediction):
    test_data = {"rowNumber": 5}
    mock_make_prediction.return_value = np.array([42])
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/prediction/", json=test_data)
    assert response.status_code == 200
    assert response.json() == {"prediction": 42}  # Assuming make_prediction returns np.array([10])
