
### Endpoints

- **GET** `/` - Service info with status, timestamp, model version, and agent load state.
```bash
curl  http://localhost:8000/ -v
```

- **GET** `/health` - Health snapshot and current model version with agent initialization status.
```bash
curl http://localhost:8000/health -v
```

- **POST** `/analyze` - Upload a fundus image (PNG/JPG) plus optional metadata to receive prediction, confidence, explanation, reasoning, governance, and heatmap_base64. You can use one of the images included in the repo inside the images folder.
```bash
curl -v -X POST http://localhost:8000/analyze \
  -F "file=@/path/to/fundus.jpg" \
  -F "age=58" \
  -F "diabetes_duration=12" \
  -F "previous_stage=1"

curl -v -X POST http://localhost:8000/analyze \
  -F "file=@/path/to/fundus.png"
```

- **GET** `/governance/logs` - Fetch recent audit records with optional filtering and limit.
```bash
curl "http://localhost:8000/governance/logs?limit=25&validated=true" -v
```

- **DELETE** `/governance/logs` - Clear in‑memory audit logs.
```bash
curl -X DELETE http://localhost:8000/governance/logs -v
```

- **GET** `/governance/statistics` - Validation rate, counts, flags, and basic usage stats.
```bash
curl  http://localhost:8000/governance/statistics -v
```

- **GET** `/agents/status` - Device and load status, reasoning mode, and per‑agent info.
```bash
curl http://localhost:8000/agents/status -v
```