from fastapi.testclient import TestClient

from app.main import app


def test_websocket_init_and_step_flow() -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws/simulate") as ws:
        ws.send_json(
            {
                "action": "init",
                "rows": 5,
                "cols": 5,
                "grid": [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            }
        )
        init_message = ws.receive_json()
        assert init_message["type"] == "init"
        assert init_message["rows"] == 5
        assert init_message["cols"] == 5

        ws.send_json({"action": "step", "mode": "cpu"})
        step_message = ws.receive_json()

        assert step_message["type"] == "step"
        assert step_message["modeUsed"] == "cpu"
        assert step_message["grid"] == [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
