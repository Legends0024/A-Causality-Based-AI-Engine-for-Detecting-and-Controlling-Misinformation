from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def run_case(name: str, payload: dict) -> None:
    response = client.post("/analyze", json=payload)
    print(f"{name}: {response.status_code}")
    print(response.json())
    print("-" * 80)


if __name__ == "__main__":
    root_response = client.get("/")
    print("health:", root_response.status_code, root_response.json())
    print("-" * 80)

    run_case(
        "text-only",
        {
            "text": "Breaking news you will not see on mainstream media, secret leaked document proves a miracle cure.",
            "k": 3,
        },
    )

    run_case(
        "text+graph",
        {
            "text": "Federal reserve raises interest rates by 25 basis points after inflation data.",
            "k": 2,
            "graph_data": {
                "root_id": "claim",
                "nodes": [
                    {"id": "claim"},
                    {"id": "user_a"},
                    {"id": "user_b"},
                    {"id": "user_c"},
                ],
                "edges": [
                    {"source": "claim", "target": "user_a"},
                    {"source": "claim", "target": "user_b"},
                    {"source": "user_b", "target": "user_c"},
                ],
            },
        },
    )
