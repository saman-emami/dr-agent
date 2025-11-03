# Ophthalmic Agentic AI System

A compact multi‑agent system that analyzes retinal fundus images for diabetic retinopathy (DR), explains its reasoning and returns a governed, audit‑ready response over FastAPI. The stack includes:
1. Vision Agent with Grad‑CAM heatmaps
2. clinical Reasoner with chain‑of‑thought
3. Governor that validates and logs every decision.

### Quick test in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saman-emami/dr-agent/blob/main/test_and_visualize.ipynb)  
For a quick test open the demo notebook in Google Colab and run all cells to load the model and analyze a few sample images.
Make sure to enable the GPU by going to Runtime → Change runtime type → Hardware accelerator → GPU, then save and reconnect before running the notebook.


### Example output
![Model Output](https://github.com/user-attachments/assets/9c399c84-5fcc-421b-882f-9190630edcb1)

```
🏥 PREDICTION:   Moderate DR
📈 CONFIDENCE:   78.7%
✓ VALIDATED:     True

REASONING:
As an expert ophthalmologist dealing with diabetic retinopathy, I need to take a closer look at this patient's situation using the Vision Analysis System's output to make sure we're assessing accurately.
First, let's consider the detected stage of diabetic retinopathy. Stage 2 means the disease has progressed, but it's not yet at the severe stage yet. This stage signifies some advanced changes compared to Stage 1, but we're not looking at severe complications like life-threatening vascular obstructions yet. It's crucial to monitor for any signs that might indicate future complications.
Next, I’ll examine the AI model confidence score, which stands at 78.7%. That's pretty significant, but I need to remember that AI scores can vary based on factors like sample size and how extensive the disease has been treated. Therefore, while the model perceives this as strong, I should still be cautious about over-relying on it without further verification.
When it comes to expected clinical features, we have microaneurysms, retinal hemorrhages, and potentially hard exudates showing up. These are definitely concerning. Microaneurysms can become dangerous soon, especially if untreated. Hard exudates signal potential obstruction because they can really mess with vision, so monitoring is essential. Retinal hemorrhages add another layer of worry; they suggest the retinal blood vessels are under stress, possibly indicating a progression to deeper issues like vasculopathy.
Considering the disease severity level, which is moderate, it’s a bit tricky. Yet, given the presence of those features, even moderate severity could warrant more attention. I need to weigh the risks and benefits here, particularly any possible progression into unseen territory that might not be obvious.
So, what about risk stratification? Progression risk seems moderate at first, but with those soft tissue changes like microaneurysms and bleeding, I'm getting a sense of progression that might not be initially apparent. Potential complications like a vascular occlusion mean immediate observation is not enough. Patient follow-ups will be critical.
There's a lot to think through. The AI score indicates reliability, but I should verify it aligns with reality. These features—microaneurysms and hemorrhages—suggest trouble that needs watching closely. Hard exudates could mean something next level important is happening, possibly even an occlusion.
In terms of management, keep in mind we’re dealing with stage 2 here. That’s where early intervention can prevent big problems. Regular visual check-ups are needed, and we should refer if things seem to be progressing too slowly or if any new features pop up. Treatment options like raising diabetes management standards or experimenting with medications like beta-blockers could be helpful. But, if hard exudates persist or lead to occlusions, urgent interventions might be necessary.
Patient education is key too. Educating them about signs that prompt follow-up and discussing potential complications will help manage their awareness and reduce their risk. This systematic approach should ensure we’re not just relying on the model's confidence but truly understanding the nuances of the patient's situation. With all of this considered, a comprehensive management plan can now be drawn up confidently.
Ultimately, while initially I thought I was on track, upon reevaluating the signs, especially with hard exudates, I must revise my earlier plan. Hard exudates demand attention due to their potential for serious complications like occlusion. This reassessment confirms the need for regular monitoring, careful follow-ups, and necessary interventions. Patients should be educated today about the importance of staying informed in managing such a condition.

EXPLANATION:
Based on the Vision Analysis System's findings and your thorough assessment, here's a structured clinical evaluation of the patient's retinal condition:
1. **Clinical Observation**: The detected stage of diabetic retinopathy is Stage 2. This indicates that the disease has progressed beyond early stages, but it remains in an intermediate phase, suggesting ongoing monitoring is necessary to prevent rapid progression or complications. 
2. **Diagnostic Confidence Assessment**: The AI model confidence score of 78.7% is significant, indicating a relatively reliable assessment. However, given the disease progression and presence of soft tissue changes like microaneurysms and retinal hemorrhages, slight inaccuracies in diagnosis might be anticipated. It's crucial to validate findings with additional data or clinical examination when necessary.
3. **Risk Stratification**: While the disease severity is classified as moderate, the presence of complications such as microaneurysms and hard exudates raises concerns about progression and potential complications. Hard exudates specifically raise suspicion for possible obstruction or underlying vascular pathology, necessitating close monitoring and follow-ups. Progression in this direction isn't adequately assessed by the AI model alone.
4. **Early Intervention and Management Options**: Given the detected stage of Stage 2, early intervention is advisable to prevent progression. Regular visual examinations are recommended to catch any developments early. Treatment considerations might include adjusting diabetes management protocols or exploring pharmacological options like beta-blockers. However, if there are
```
### Features

- **Multi‑agent** design: Vision, Reasoner, and Governor coordinated by a lightweight ReAct‑style orchestrator for validated outputs.
- **DR staging**: 5 classes (0–4) mapped to clinically meaningful stage names and severities used consistently across modules.
- **Explainability**: Chain‑of‑thought reasoning with an LLM mode and a deterministic rule‑based fallback; final summary is separated from the chain.
- **Governance**: Confidence/consistency checks, timestamps, model versioning, trace IDs, flags, and in‑memory audit logs with query and maintenance endpoints.
- **Grad‑CAM**: Heatmap highlighting key retinal regions.


### Architecture

- **VisionAgent**: Hugging Face classifier for DR, preprocessing, prediction, and Grad‑CAM; returns image_id, stage, confidence, and heatmap.
- **ReasonerAgent**: Produces a clinical explanation plus stepwise reasoning; supports medical LLM CoT and a rule‑based path with clear recommendations.
- **GovernorAgent**: Validates ranges and thresholds, checks cross‑agent consistency, emits an audit‑ready response with flags and rule outcomes.
- **ReactOrchestrator**: Executes vision once, iterates reasoning with validation, then falls back to rules if needed; returns the first validated result or a failure trace.
- **FastAPI app**: Unified entrypoint exposing health, analysis, governance logs/statistics, and agent status.


## Docker Compose quick commands

Build the image:

```bash
docker compose build
```

Start the stack:

```bash
docker compose up -d
```

## API

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

## Security and Clinical Use

This software is for research and prototyping; it is not a medical device. Use only in controlled settings, with clinician oversight, and do not make clinical decisions solely on its output.

## License and Attribution

Ensure compliance with licenses for third‑party models and datasets before any production integration. Include appropriate attributions and license files as required.
