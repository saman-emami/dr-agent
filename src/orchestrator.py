"""
react_orchestrator.py

Simplified ReAct orchestration system for coordinating Vision, Reasoner,
and Governor agents with iterative validation and rule-based fallback.
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Configure logger for module
logger = logging.getLogger("Orchestrator")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[Orchestrator] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class ReactExecutionResult:
    """
    Complete result of ReAct orchestration execution.

    Attributes
    ----------
    session_id : str
        Unique identifier for this execution session.
    success : bool
        Whether a validated result was achieved.
    final_result : dict or None
        Final governed output if successful.
    total_iterations : int
        Total number of iterations performed.
    fallback_activated : bool
        Whether rule-based fallback was triggered.
    execution_time_seconds : float
        Total execution time in seconds.
    error_message : str or None
        Error description if execution failed.
    """

    session_id: str
    success: bool
    final_result: Optional[Dict[str, Any]]
    total_iterations: int
    fallback_activated: bool
    execution_time_seconds: float
    error_message: Optional[str] = None


class ReactOrchestrator:
    """
    Simplified ReAct orchestration layer coordinating Vision, Reasoner, and Governor.

    Implements iterative reasoning-validation loop with automatic fallback:
    1. Execute Vision once (cached)
    2. Iterate with LLM-based reasoning + validation
    3. If max retries reached, fallback to rule-based reasoning
    4. Return final validated result or failure

    Parameters
    ----------
    vision_agent : VisionAgent
        Vision agent for image analysis.
    reasoner_agent : ReasonerAgent
        Reasoner agent for explanation generation.
    governor_agent : GovernorAgent
        Governor agent for validation.
    maximum_llm_retries : int, optional
        Max LLM reasoning attempts (default: 3).
    maximum_rule_retries : int, optional
        Max rule-based fallback attempts (default: 2).
    timeout_seconds : float, optional
        Max execution time (default: 120.0).
    """

    def __init__(
        self,
        vision_agent,
        reasoner_agent,
        governor_agent,
        maximum_llm_retries: int = 3,
        timeout_seconds: float = 120.0,
    ):
        """Initialize orchestrator with agents and configuration."""
        self.vision_agent = vision_agent
        self.reasoner_agent = reasoner_agent
        self.governor_agent = governor_agent
        self.maximum_llm_retries = maximum_llm_retries
        self.timeout_seconds = timeout_seconds

        logger.info(f"ReactOrchestrator initialized: llm_retries={maximum_llm_retries}")

    def execute(
        self,
        image_source: Any,
        patient_metadata: Optional[Dict[str, Any]] = None,
    ) -> ReactExecutionResult:
        """
        Execute complete ReAct orchestration with iterative validation.

        Workflow:
        1. Execute VisionAgent once (cached for iterations)
        2. Phase 1: Try LLM-based reasoning up to maximum_llm_retries
        3. Phase 2: If unsuccessful, try rule-based reasoning up to maximum_rule_retries
        4. Return validated result or failure

        Parameters
        ----------
        image_source : Any
            Input image (file path, numpy array, or PIL Image).
        patient_metadata : dict, optional
            Patient context (age, diabetes_duration, previous_stage).

        Returns
        -------
        ReactExecutionResult
            Complete execution result with success status and trace.
        """
        session_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(f"Starting ReAct orchestration - Session: {session_id}")

        # Execute vision once and cache
        vision_output = self.vision_agent.predict(image_source)
        logger.info(
            f"Vision: stage={vision_output.get('stage')}, "
            f"confidence={vision_output.get('confidence', 0):.2%}"
        )

        total_iterations = 0
        fallback_activated = False
        final_result = None
        error_message = None

        # Phase 1: LLM-based reasoning
        self.reasoner_agent.switch_reasoning_mode(use_llm=True)
        logger.info(f"Phase 1: LLM reasoning (max {self.maximum_llm_retries} attempts)")
        for iteration in range(1, self.maximum_llm_retries + 1):
            if time.time() - start_time > self.timeout_seconds:
                error_message = "Timeout exceeded"
                break

            total_iterations += 1
            logger.info(f"Iteration {iteration}: LLM reasoning")

            try:
                reasoning_output = self.reasoner_agent.reason(
                    vision_output=vision_output,
                    patient_metadata=patient_metadata,
                )

                # Validate with governor
                governance_result = self.governor_agent.govern(
                    vision_output=vision_output,
                    reasoning_output=reasoning_output,
                )

                validated = governance_result.get("governance", {}).get(
                    "validated", False
                )

                if validated:
                    final_result = governance_result
                    logger.info(f"Validation passed on iteration {iteration}")
                    break
                else:
                    logger.warning(f"Validation failed on iteration {iteration}")

            except Exception as error:
                logger.error(f"Iteration {iteration} error: {error}")

        # Phase 2: Rule-based fallback if unsuccessful
        if final_result is None and error_message is None:
            self.reasoner_agent.switch_reasoning_mode(use_llm=False)
            print("switch to rule based....")
            logger.info("Phase 2: Single rule-based fallback")
            fallback_activated = True
            total_iterations += 1

            self.reasoner_agent.use_llm = False
            reasoning_output = self.reasoner_agent.reason(
                vision_output=vision_output, patient_metadata=patient_metadata
            )
            governance_result = self.governor_agent.govern(
                vision_output=vision_output, reasoning_output=reasoning_output
            )
            validated = governance_result.get("governance", {}).get("validated", False)

            if validated:
                final_result = governance_result
                logger.info("✓ Fallback (rule-based) validation passed")
            else:
                logger.warning("✗ Fallback (rule-based) validation failed")
                error_message = (
                    "All LLM attempts and the single rule-based fallback failed"
                )

        # Finalize result
        execution_time = time.time() - start_time
        success = final_result is not None

        result = ReactExecutionResult(
            session_id=session_id,
            success=success,
            final_result=final_result,
            total_iterations=total_iterations,
            fallback_activated=fallback_activated,
            execution_time_seconds=round(execution_time, 3),
            error_message=error_message,
        )

        logger.info(
            f"Orchestration complete: success={success}, "
            f"iterations={total_iterations}, time={execution_time:.2f}s"
        )

        return result
