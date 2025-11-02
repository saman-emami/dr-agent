"""
app.py

FastAPI application for Ophthalmic Agentic AI System.
"""

import io
import base64
import logging
from typing import Optional, Dict, Any, Union
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import uvicorn

from .vision_agent import VisionAgent
from .reasoner_agent import ReasonerAgent
from .governor_agent import GovernorAgent
from .orchestrator import ReactOrchestrator

# Configure logger for module
logger = logging.getLogger("OphthalmicAPI")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[OphthalmicAPI] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class PatientMetadata(BaseModel):
    """
    Optional patient metadata for enhanced clinical reasoning.

    Attributes
    ----------
    age : int, optional
        Patient age in years (must be between 0 and 120).
    diabetes_duration : int, optional
        Duration of diabetes diagnosis in years.
    previous_stage : int, optional
        Previously diagnosed DR stage (0-4).
    """

    age: Optional[int] = Field(None, ge=0, le=120, description="Patient age in years")
    diabetes_duration: Optional[int] = Field(
        None, ge=0, description="Diabetes duration in years"
    )
    previous_stage: Optional[int] = Field(
        None, ge=0, le=4, description="Previous DR stage"
    )


class AnalysisResponse(BaseModel):
    """
    Complete analysis response structure for retinal image assessment.

    Attributes
    ----------
    image_id : str
        Unique identifier for the analyzed image.
    prediction : str
        Final DR stage prediction (e.g., "Moderate DR").
    confidence : float
        Model confidence score (0.0 to 1.0).
    explanation : str
        Human-readable clinical explanation.
    reasoning : str
        Chain-of-thought reasoning steps.
    governance : dict
        Governance and audit metadata.
    heatmap_base64 : str, optional
        Base64-encoded Grad-CAM heatmap visualization.
    """

    image_id: str
    prediction: str
    confidence: float
    explanation: str
    reasoning: str
    governance: Dict[str, Any]
    heatmap_base64: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """
    API health status response.

    Attributes
    ----------
    status : str
        Service operational status.
    timestamp : str
        Current ISO 8601 timestamp.
    model_version : str
        Deployed model version identifier.
    agents_loaded : bool
        Whether all agents are successfully initialized.
    """

    status: str
    timestamp: str
    model_version: str
    agents_loaded: bool


vision_agent: Optional[VisionAgent] = None
reasoner_agent: Optional[ReasonerAgent] = None
governor_agent: Optional[GovernorAgent] = None
orchestrator: Optional[ReactOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown.

    Startup: Initialize all agents before the app starts receiving requests.
    Shutdown: Clean up resources after the app finishes.
    """
    global vision_agent, reasoner_agent, governor_agent, orchestrator

    # Startup logic
    try:
        logger.info("Initializing Ophthalmic Agentic AI System...")

        logger.info("Loading Vision Agent...")
        vision_agent = VisionAgent()

        logger.info("Loading Reasoner Agent...")
        reasoner_agent = ReasonerAgent()

        logger.info("Loading Governor Agent...")
        governor_agent = GovernorAgent()

        logger.info("Initializing ReAct Orchestrator...")
        orchestrator = ReactOrchestrator(
            vision_agent=vision_agent,
            reasoner_agent=reasoner_agent,
            governor_agent=governor_agent,
        )

        logger.info("All agents initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        raise

    yield  # Application runs here

    # Shutdown logic
    logger.info("Shutting down Ophthalmic Agentic AI System...")
    try:
        vision_agent = None
        reasoner_agent = None
        governor_agent = None
        orchestrator = None
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


app = FastAPI(
    title="Ophthalmic Agentic AI System",
    description="Multi-agent system for diabetic retinopathy detection with AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def numpy_to_base64(array: np.ndarray) -> Union[str, None]:
    """
    Convert numpy array to base64-encoded PNG string.

    Parameters
    ----------
    array : numpy.ndarray
        Input image array (RGB format).

    Returns
    -------
    str
        Base64-encoded data URI string, or None if conversion fails.
    """
    try:
        # Convert to uint8 if necessary
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)

        # Convert to PIL Image
        image = Image.fromarray(array)

        # Encode as base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        base64_str = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_str}"

    except Exception as e:
        logger.error(f"Failed to convert heatmap to base64: {e}")
        return None


async def process_uploaded_image(file: UploadFile) -> Image.Image:
    """
    Process uploaded image file and convert to PIL Image.

    Parameters
    ----------
    file : UploadFile
        Uploaded image file from FastAPI request.

    Returns
    -------
    PIL.Image.Image
        Processed RGB image.

    Raises
    ------
    HTTPException
        If image processing fails or file format is invalid.
    """
    try:
        # Read file content
        contents = await file.read()

        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    except Exception as e:
        logger.error(f"Failed to process uploaded image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")


@app.get("/", response_model=HealthCheckResponse)
async def root():
    """
    Root endpoint providing basic system information.

    Returns
    -------
    HealthCheckResponse
        System status and metadata.
    """
    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_version": "v1.0",
        "agents_loaded": all(
            [vision_agent, reasoner_agent, governor_agent, orchestrator]
        ),
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Detailed health check endpoint for monitoring.

    Returns
    -------
    HealthCheckResponse
        Detailed system health status including agent initialization state.
    """
    agents_loaded = all([vision_agent, reasoner_agent, governor_agent, orchestrator])

    return {
        "status": "healthy" if agents_loaded else "degraded",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_version": governor_agent.model_version if governor_agent else "unknown",
        "agents_loaded": agents_loaded,
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_retina_image(
    file: UploadFile = File(..., description="Retinal fundus image (JPG, PNG)"),
    age: Optional[int] = Form(None, description="Patient age"),
    diabetes_duration: Optional[int] = Form(
        None, description="Diabetes duration in years"
    ),
    previous_stage: Optional[int] = Form(None, description="Previous DR stage (0-4)"),
):
    """
    Main analysis endpoint for diabetic retinopathy detection.

    Accepts retinal fundus image with optional patient metadata and returns
    complete analysis including prediction, confidence, explanation, reasoning,
    governance audit trail, and Grad-CAM heatmap visualization.

    Parameters
    ----------
    file : UploadFile
        Retinal fundus image file (JPEG or PNG format).
    age : int, optional
        Patient age in years.
    diabetes_duration : int, optional
        Duration of diabetes in years.
    previous_stage : int, optional
        Previously diagnosed DR stage (0-4).

    Returns
    -------
    AnalysisResponse
        Complete analysis with prediction, confidence, explanation, reasoning,
        governance data, and heatmap visualization.

    Raises
    ------
    HTTPException
        - 503: If agents are not initialized
        - 400: If image file is invalid
        - 500: If analysis fails
    """

    # Verify agents are initialized
    if not all([vision_agent, reasoner_agent, governor_agent, orchestrator]):
        raise HTTPException(
            status_code=503, detail="AI agents not initialized. Please try again later."
        )

    try:
        # Process uploaded image
        logger.info(f"Processing uploaded image: {file.filename}")
        image = await process_uploaded_image(file)

        # Prepare patient metadata
        patient_metadata = None
        if any([age, diabetes_duration, previous_stage is not None]):
            patient_metadata = {
                "age": age,
                "diabetes_duration": diabetes_duration,
                "previous_stage": previous_stage,
            }
            logger.info(f"Patient metadata provided: {patient_metadata}")

        # Execute orchestrated analysis
        logger.info("Starting orchestrated analysis...")
        result = orchestrator.execute(
            image_source=image, patient_metadata=patient_metadata
        )

        # Check if analysis was successful
        if not result.success:
            raise HTTPException(
                status_code=500, detail=f"Analysis failed: {result.error_message}"
            )

        # Extract final result
        final_result = result.final_result

        # Convert heatmap to base64
        heatmap_base64 = None
        if "key_regions" in final_result:
            heatmap_base64 = numpy_to_base64(final_result["key_regions"])

        # Construct response
        response = AnalysisResponse(
            image_id=final_result["image_id"],
            prediction=final_result["prediction"],
            confidence=final_result["confidence"],
            explanation=final_result["explanation"],
            reasoning=final_result.get("reasoning", ""),
            governance=final_result["governance"],
            heatmap_base64=heatmap_base64,
        )

        logger.info(
            f"Analysis complete - Image: {response.image_id}, "
            f"Prediction: {response.prediction}, "
            f"Validated: {response.governance.get('validated', False)}"
        )

        return response

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Analysis failed with error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal server error during analysis: {str(e)}"
        )


@app.get("/governance/logs")
async def get_governance_logs(
    limit: Optional[int] = 50, validated: Optional[bool] = None
):
    """
    Retrieve audit trail logs from governance system.

    Parameters
    ----------
    limit : int, optional
        Maximum number of records to return (default: 50).
    validated : bool, optional
        Filter by validation status (true/false/null for all).

    Returns
    -------
    dict
        Contains:
        - total_records : int
            Number of records returned.
        - filter_applied : dict
            Applied filter parameters.
        - logs : list of dict
            Audit records with timestamps, predictions, and validation results.

    Raises
    ------
    HTTPException
        - 503: If governor agent is not initialized
        - 500: If log retrieval fails
    """

    if not governor_agent:
        raise HTTPException(status_code=503, detail="Governor agent not initialized")

    try:
        logs = governor_agent.get_audit_logs(limit=limit, filter_validated=validated)

        return {
            "total_records": len(logs),
            "filter_applied": {"limit": limit, "validated": validated},
            "logs": logs,
        }

    except Exception as e:
        logger.error(f"Failed to retrieve audit logs: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve audit logs: {str(e)}"
        )


@app.get("/governance/statistics")
async def get_governance_statistics():
    """
    Retrieve governance statistics and performance metrics.

    Returns
    -------
    dict
        Statistics including:
        - total_analyses : int
            Total number of analyses performed.
        - validation_rate : float
            Proportion of analyses that passed validation.
        - most_common_flags : list
            Most frequently triggered flags.
        - model_version : str
            Current model version.
        - storage_utilization : str
            Audit storage usage.

    Raises
    ------
    HTTPException
        - 503: If governor agent is not initialized
        - 500: If statistics retrieval fails
    """

    if not governor_agent:
        raise HTTPException(status_code=503, detail="Governor agent not initialized")

    try:
        stats = governor_agent.get_governance_statistics()
        return stats

    except Exception as e:
        logger.error(f"Failed to retrieve statistics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve statistics: {str(e)}"
        )


@app.delete("/governance/logs")
async def clear_audit_logs():
    """
    Clear all audit logs.

    Returns
    -------
    dict
        Confirmation message.

    Raises
    ------
    HTTPException
        - 503: If governor agent is not initialized
        - 500: If clearing fails
    """

    if not governor_agent:
        raise HTTPException(status_code=503, detail="Governor agent not initialized")

    try:
        governor_agent.clear_audit_logs()
        return {"message": "Audit logs cleared successfully"}

    except Exception as e:
        logger.error(f"Failed to clear audit logs: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to clear audit logs: {str(e)}"
        )


@app.get("/agents/status")
async def get_agents_status():
    """
    Get detailed status information for all agents.

    Returns
    -------
    dict
        Status information for Vision, Reasoner, Governor agents and Orchestrator,
        including:
        - loaded : bool
            Whether agent is initialized.
        - device : str
            Compute device (CPU/CUDA).
        - mode : str
            Operating mode for reasoner.
        - statistics : dict
            Performance metrics.
    """

    return {
        "vision_agent": {
            "loaded": vision_agent is not None,
            "device": str(vision_agent.device) if vision_agent else None,
        },
        "reasoner_agent": {
            "loaded": reasoner_agent is not None,
            "mode": (
                "llm_medical_cot"
                if (reasoner_agent and reasoner_agent.use_llm)
                else "rule_based"
            ),
            "device": str(reasoner_agent.device) if reasoner_agent else None,
            "statistics": (
                reasoner_agent.get_reasoning_statistics() if reasoner_agent else None
            ),
        },
        "governor_agent": {
            "loaded": governor_agent is not None,
            "model_version": governor_agent.model_version if governor_agent else None,
            "statistics": (
                governor_agent.get_governance_statistics() if governor_agent else None
            ),
        },
        "orchestrator": {"loaded": orchestrator is not None},
    }
