from src import get_season_result


def test_season_ingestion_reads_all_result_pages(monkeypatch):
    def race(round_number, drivers):
        return {
            "season": "2024",
            "round": str(round_number),
            "date": "2024-03-02",
            "Circuit": {"circuitId": "test"},
            "Results": [
                {
                    "Driver": {"driverId": driver},
                    "Constructor": {"constructorId": "team"},
                    "grid": str(index + 1),
                    "position": str(index + 1),
                    "status": "Finished",
                }
                for index, driver in enumerate(drivers)
            ],
        }

    pages = {
        0: {
            "MRData": {
                "limit": "2",
                "total": "3",
                "RaceTable": {"Races": [race(1, ["a", "b"])]},
            }
        },
        2: {
            "MRData": {
                "limit": "2",
                "total": "3",
                "RaceTable": {"Races": [race(1, ["c"])]},
            }
        },
    }
    offsets = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, params, timeout):
        offsets.append(params["offset"])
        return Response(pages[params["offset"]])

    monkeypatch.setattr("src.requests.get", fake_get)
    results = get_season_result(2024)

    assert offsets == [0, 2]
    assert len(results) == 3
    assert results["driver"].tolist() == ["a", "b", "c"]
