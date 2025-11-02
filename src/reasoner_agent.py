"""
reasoner_agent.py

A self-contained class for generating explainable clinical reasoning summaries
for diabetic retinopathy detection using Qwen-Distilled-Scout-1.5B-Instruct-Gen2.
This model is specifically fine-tuned for medical chain-of-thought reasoning.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Configure logger for module
logger = logging.getLogger("ReasonerAgent")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[ReasonerAgent] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for generation to stop at specific tokens."""

    def __init__(self, stop_token_ids: list):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        return any(
            input_ids[0, -len(token) :].tolist() == token
            for token in self.stop_token_ids
        )


@dataclass
class ReasoningStep:
    """
    Structured representation of a single reasoning step.

    Attributes
    ----------
    step_number : int
        Sequential identifier for the reasoning step.
    observation : str
        What the agent observes from the input data.
    analysis : str
        The agent's interpretation of the observation.
    conclusion : str
        Intermediate or final conclusion drawn from the analysis.
    """

    step_number: int
    observation: str
    analysis: str
    conclusion: str


class ReasonerAgent:
    """
    ReasonerAgent for generating clinical explanations of DR detection results.
    Uses Qwen-Distilled-Scout-1.5B-Instruct-Gen2 model fine-tuned specifically
    for medical chain-of-thought reasoning.

    Parameters
    ----------
    model_id : str, optional
        Hugging Face model identifier (default: "eagle0504/qwen-distilled-scout-1.5b-instruct-gen2").
    use_llm : bool, optional
        If True, use LLM for reasoning; if False, use rule-based method (default: True).
    enable_caching : bool, optional
        Enable model caching for faster repeated inference (default: True).
    """

    # Clinical thresholds and parameters
    CONFIDENCE_THRESHOLDS = {"high": 0.85, "moderate": 0.65, "low": 0.50}

    STAGE_DEFINITIONS = {
        0: {
            "name": "No DR",
            "description": "no signs of diabetic retinopathy",
            "clinical_features": "normal retinal appearance with intact vasculature",
            "severity": "none",
        },
        1: {
            "name": "Mild DR",
            "description": "mild non-proliferative diabetic retinopathy",
            "clinical_features": "microaneurysms only",
            "severity": "minimal",
        },
        2: {
            "name": "Moderate DR",
            "description": "moderate non-proliferative diabetic retinopathy",
            "clinical_features": "microaneurysms, retinal hemorrhages, and possibly hard exudates",
            "severity": "moderate",
        },
        3: {
            "name": "Severe DR",
            "description": "severe non-proliferative diabetic retinopathy",
            "clinical_features": "extensive intraretinal hemorrhages, venous beading, and intraretinal microvascular abnormalities",
            "severity": "severe",
        },
        4: {
            "name": "Proliferative DR",
            "description": "proliferative diabetic retinopathy",
            "clinical_features": "neovascularization, vitreous hemorrhage, or preretinal hemorrhage",
            "severity": "critical",
        },
    }

    def __init__(
        self,
        model_id: str = "eagle0504/qwen-distilled-scout-1.5b-instruct-gen2",
        use_llm: bool = True,
        enable_caching: bool = True,
    ):
        self.use_llm = use_llm
        self.enable_caching = enable_caching
        self.reasoning_cache = {} if enable_caching else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id

        # Initialize LLM components if needed
        if self.use_llm:
            logger.info(f"Initializing medical CoT reasoning with model: {model_id}")
            logger.info(f"Using device: {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=(torch.float16 if self.device.type == "cuda" else torch.float32),
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()

            # Set up stopping criteria for response termination
            stop_sequence = "</response>"
            stop_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
            self.stopping_criteria = StoppingCriteriaList([StopOnTokens([stop_ids])])

            logger.info(
                "Qwen-Distilled-Scout medical reasoning model loaded successfully"
            )
        else:
            logger.info("Using rule-based reasoning mode")
            self.tokenizer = None
            self.model = None
            self.stopping_criteria = None

    def _construct_medical_cot_prompt(
        self,
        vision_output: Dict[str, Any],
        patient_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build a structured prompt following medical CoT best practices for the
        Qwen-Distilled-Scout-1.5B-Instruct-Gen2 model.

        This follows the model's expected format:
        <instruction>This is a medical problem.</instruction>
        <question>...</question>

        The model will generate:
        <think>...</think>
        <response>...</response>

        Parameters
        ----------
        vision_output : dict
            Output from VisionAgent containing stage, confidence, key_regions.
        patient_metadata : dict, optional
            Additional patient information (age, diabetes_duration, previous_stage).

        Returns
        -------
        str
            Formatted prompt for LLM inference.
        """
        stage = vision_output.get("stage", -1)
        confidence = vision_output.get("confidence", 0.0)
        stage_info = self.STAGE_DEFINITIONS.get(
            stage,
            {
                "name": "Unknown",
                "description": "unknown stage",
                "clinical_features": "unclear findings",
            },
        )

        # Build patient context string
        patient_context = ""
        if patient_metadata:
            patient_details = []
            if patient_metadata.get("age"):
                patient_details.append(f"Age: {patient_metadata['age']} years")
            if patient_metadata.get("diabetes_duration"):
                patient_details.append(
                    f"Diabetes duration: {patient_metadata['diabetes_duration']} years"
                )
            if patient_metadata.get("previous_stage") is not None:
                prev_stage_info = self.STAGE_DEFINITIONS.get(
                    patient_metadata["previous_stage"], {}
                )
                patient_details.append(
                    f"Previous diagnosis: {prev_stage_info.get('name', 'Unknown')}"
                )

            if patient_details:
                patient_context = "\nPatient History: " + ", ".join(patient_details)

        # Construct the medical CoT prompt following the model's training format
        prompt = f"""<instruction>This is a medical problem.</instruction>
<question>As an expert ophthalmologist specializing in diabetic retinopathy, analyze the following retinal image assessment results and provide a comprehensive clinical evaluation.

Clinical Data from Vision Analysis System:
- Detected Stage: Stage {stage} - {stage_info['name']}
- AI Model Confidence Score: {confidence:.1%}
- Expected Clinical Features: {stage_info['clinical_features']}
- Disease Severity Level: {stage_info['severity']}{patient_context}

Clinical Task:
Perform a systematic chain-of-thought medical reasoning analysis to evaluate this diabetic retinopathy case. Your analysis must follow evidence-based ophthalmology practice and and in just 3-4 sentences provide:

1. Clinical Observation: Interpret what the detected stage indicates about the patient's retinal pathology and the significance of identified features.

2. Diagnostic Confidence Assessment: Evaluate the reliability of the AI prediction based on the confidence score, considering factors that may affect accuracy and any need for additional validation.

3. Risk Stratification: Assess the progression risk, potential complications, and urgency of intervention based on the severity level and patient history.

4. Evidence-Based Recommendations: Provide specific clinical management recommendations including follow-up timeline, referral requirements, treatment considerations, and patient education points.

Please structure your response with clear step-by-step reasoning that demonstrates clinical judgment and supports the final recommendations.</question>"""

        return prompt

    def _perform_llm_inference(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """
        Execute LLM inference using Qwen-Distilled-Scout model.

        Parameters
        ----------
        prompt : str
            Input prompt for the model.
        max_new_tokens : int, optional
            Maximum tokens to generate (default: 1024).

        Returns
        -------
        str
            Generated reasoning text.
        """
        # Tokenize input
        input_tokens = self.tokenizer(
            prompt, return_tensors="pt", max_length=1024, truncation=True
        ).to(self.device)

        # Generate with optimized parameters for medical reasoning
        with torch.no_grad():
            output_tokens = self.model.generate(
                **input_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.4,  # Lower temperature for more focused medical reasoning
                top_p=0.85,
                top_k=40,
                repetition_penalty=1.1,
                stopping_criteria=self.stopping_criteria,
            )

        # Decode output
        generated_text = self.tokenizer.decode(
            output_tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return generated_text

    def _parse_model_output(self, generated_text: str) -> Dict[str, str]:
        """
        Parse the model's output to extract reasoning (<think>) and explanation (<response>).

        Parameters
        ----------
        generated_text : str
            The complete generated text from the model.

        Returns
        -------
        dict
            Dictionary with 'reasoning' and 'explanation' keys.
        """
        # Extract the <think> section (reasoning)
        think_match = re.search(r"<think>(.*?)</think>", generated_text, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else ""

        # Extract the <response> section (final explanation)
        response_match = re.search(
            r"<response>(.*?)(?:</response>|$)", generated_text, re.DOTALL
        )
        explanation = response_match.group(1).strip() if response_match else ""

        # If no structured output found, try to parse differently
        if not reasoning and not explanation:
            # The model might have output without tags
            parts = generated_text.split("<question>")
            if len(parts) > 1:
                output = parts[-1].strip()
                # Try to identify reasoning vs explanation by content
                lines = output.split("\n\n")
                if len(lines) > 1:
                    reasoning = "\n\n".join(lines[:-1])
                    explanation = lines[-1]
                else:
                    explanation = output

        return {"reasoning": reasoning, "explanation": explanation}

    def _execute_rule_based_reasoning(
        self,
        vision_output: Dict[str, Any],
        patient_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ReasoningStep]:
        """
        Generate reasoning using deterministic clinical rules.

        Parameters
        ----------
        vision_output : dict
            Output from VisionAgent.
        patient_metadata : dict, optional
            Additional patient context.

        Returns
        -------
        list of ReasoningStep
            Structured reasoning chain.
        """
        stage = vision_output.get("stage", -1)
        confidence = vision_output.get("confidence", 0.0)
        stage_info = self.STAGE_DEFINITIONS.get(stage, self.STAGE_DEFINITIONS[0])

        reasoning_steps = []

        # Step 1: Observation of retinal findings
        observation_text = (
            f"The vision model detected {stage_info['name']} with "
            f"{stage_info['clinical_features']}. "
        )

        if confidence >= self.CONFIDENCE_THRESHOLDS["high"]:
            confidence_level = "high confidence"
        elif confidence >= self.CONFIDENCE_THRESHOLDS["moderate"]:
            confidence_level = "moderate confidence"
        else:
            confidence_level = "low confidence requiring verification"

        observation_text += f"The detection has {confidence_level} ({confidence:.1%})."

        step_one = ReasoningStep(
            step_number=1,
            observation=observation_text,
            analysis=f"Stage {stage} indicates {stage_info['description']}",
            conclusion=f"Primary diagnosis: {stage_info['name']} with {stage_info['severity']} severity",
        )

        reasoning_steps.append(step_one)

        # Step 2: Confidence and reliability assessment
        reliability_factors = []

        if confidence >= self.CONFIDENCE_THRESHOLDS["high"]:
            reliability_assessment = (
                "The high confidence score suggests robust feature detection"
            )
            recommendation_confidence = "can be used for clinical decision support"
        elif confidence >= self.CONFIDENCE_THRESHOLDS["moderate"]:
            reliability_assessment = (
                "The moderate confidence score indicates reasonable certainty"
            )
            recommendation_confidence = (
                "should be reviewed by a clinician before final diagnosis"
            )
        else:
            reliability_assessment = (
                "The low confidence score suggests ambiguous findings"
            )
            recommendation_confidence = (
                "requires expert verification and possibly additional imaging"
            )

        # Consider patient history if available
        if (
            patient_metadata
            and patient_metadata.get("previous_stage") is not None
            and patient_metadata.get("previous_stage") != stage
        ):
            previous_stage = patient_metadata["previous_stage"]
            stage_difference = abs(stage - previous_stage)
            if stage_difference > 1:
                reliability_factors.append(
                    f"significant change from previous stage {previous_stage} warrants careful review"
                )

        reliability_text = reliability_assessment
        if reliability_factors:
            reliability_text += ". Additionally, " + ", and ".join(reliability_factors)

        step_two = ReasoningStep(
            step_number=2,
            observation=f"Prediction confidence is {confidence:.1%}",
            analysis=reliability_text,
            conclusion=f"This result {recommendation_confidence}",
        )

        reasoning_steps.append(step_two)

        # Step 3: Clinical interpretation and actionable guidance
        clinical_guidance = self._generate_clinical_recommendations(
            stage, confidence, patient_metadata
        )

        step_three = ReasoningStep(
            step_number=3,
            observation=f"Clinical context: {stage_info['severity']} severity diabetic retinopathy",
            analysis=clinical_guidance["rationale"],
            conclusion=clinical_guidance["recommendation"],
        )

        reasoning_steps.append(step_three)

        return reasoning_steps

    def _generate_clinical_recommendations(
        self,
        stage: int,
        confidence: float,
        patient_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Create evidence-based clinical recommendations.

        Parameters
        ----------
        stage : int
            DR severity stage (0-4).
        confidence : float
            Prediction confidence score.
        patient_metadata : dict, optional
            Patient history and context.

        Returns
        -------
        dict
            Contains 'rationale' and 'recommendation' keys.
        """
        recommendations = {
            0: {
                "rationale": "No retinopathy detected; routine screening maintains early detection capability",
                "recommendation": "Continue annual diabetic eye examinations and optimize glycemic control",
            },
            1: {
                "rationale": "Mild changes indicate early microvascular damage requiring monitoring",
                "recommendation": "Schedule follow-up in 6-12 months; reinforce diabetes management",
            },
            2: {
                "rationale": "Moderate changes show progressive retinal damage with risk of advancement",
                "recommendation": "Refer to ophthalmology for detailed examination within 2-4 weeks; consider more frequent monitoring",
            },
            3: {
                "rationale": "Severe non-proliferative changes carry high risk of progression to vision-threatening complications",
                "recommendation": "Urgent ophthalmology referral within 1 week; likely candidate for preventive laser photocoagulation",
            },
            4: {
                "rationale": "Proliferative disease with neovascularization poses immediate risk of severe vision loss",
                "recommendation": "Emergent ophthalmology consultation within 24-48 hours; requires prompt treatment with laser or anti-VEGF therapy",
            },
        }

        base_recommendation = recommendations.get(stage, recommendations[0])

        # Adjust for low confidence
        if confidence < self.CONFIDENCE_THRESHOLDS["moderate"]:
            rationale = (
                base_recommendation["rationale"]
                + ". However, given detection uncertainty, additional validation is essential"
            )
            recommendation = (
                "Recommend repeat imaging and ophthalmologist review to confirm findings. "
                + base_recommendation["recommendation"]
            )
        else:
            rationale = base_recommendation["rationale"]
            recommendation = base_recommendation["recommendation"]

        # Adjust for patient history
        if patient_metadata:
            diabetes_duration = patient_metadata.get("diabetes_duration", 0)
            if diabetes_duration > 15:
                recommendation += " Given long diabetes duration, ensure comprehensive retinal evaluation"

        return {"rationale": rationale, "recommendation": recommendation}

    def _format_reasoning_steps_to_text(
        self, reasoning_steps: List[ReasoningStep]
    ) -> str:
        """
        Convert structured reasoning steps into readable clinical text.

        Parameters
        ----------
        reasoning_steps : list of ReasoningStep
            Chain of reasoning steps.

        Returns
        -------
        str
            Formatted explanation text.
        """
        output_lines = []

        for step in reasoning_steps:
            output_lines.append(f"\nStep {step.step_number} - Observation:")
            output_lines.append(step.observation)
            output_lines.append(f"\nAnalysis:")
            output_lines.append(step.analysis)
            output_lines.append(f"\nConclusion:")
            output_lines.append(step.conclusion)

        # Add final synthesis
        final_step = reasoning_steps[-1]
        output_lines.append("\n" + "=" * 60)
        output_lines.append("Clinical Summary:")
        output_lines.append(final_step.conclusion)

        return "\n".join(output_lines)

    def _format_reasoning_steps_for_separation(
        self, reasoning_steps: List[ReasoningStep]
    ) -> Dict[str, str]:
        """
        Format reasoning steps into separated reasoning and explanation components.

        Parameters
        ----------
        reasoning_steps : list of ReasoningStep
            Chain of reasoning steps.

        Returns
        -------
        dict
            Dictionary with 'reasoning' and 'explanation' keys.
        """
        reasoning_lines = []

        # Compile all reasoning steps
        for step in reasoning_steps:
            reasoning_lines.append(f"Step {step.step_number} - {step.observation}")
            reasoning_lines.append(f"Analysis: {step.analysis}")
            reasoning_lines.append("")

        reasoning_text = "\n".join(reasoning_lines).strip()

        # The final step's conclusion serves as the explanation
        final_step = reasoning_steps[-1]
        explanation_text = final_step.conclusion

        return {"reasoning": reasoning_text, "explanation": explanation_text}

    def reason(
        self,
        vision_output: Dict[str, Any],
        patient_metadata: Optional[Dict[str, Any]] = None,
        return_structured: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate explainable reasoning summary from vision analysis results.

        Parameters
        ----------
        vision_output : dict
            Output from VisionAgent containing:
            - stage: int (DR severity level)
            - confidence: float (prediction confidence)
            - key_regions: numpy.ndarray (Grad-CAM heatmap)
            - image_id: str (unique identifier)
        patient_metadata : dict, optional
            Additional context such as:
            - age: int
            - diabetes_duration: int (years)
            - previous_stage: int
        return_structured : bool, optional
            If True, return structured ReasoningStep objects;
            if False, return formatted text (default: False).

        Returns
        -------
        dict
            {
                'image_id': str,
                'reasoning': str (chain-of-thought steps),
                'explanation': str (final clinical summary),
                'reasoning_mode': str,
                'stage_name': str,
                'severity_level': str,
                'clinical_recommendation': str,
                'confidence_level': str
            }
        """
        # Validate input
        if "stage" not in vision_output or "confidence" not in vision_output:
            raise ValueError("vision_output must contain 'stage' and 'confidence' keys")

        stage = vision_output["stage"]
        confidence = vision_output["confidence"]
        image_id = vision_output.get("image_id", "unknown")

        # Check cache if enabled
        if self.enable_caching and self.reasoning_cache is not None:
            cache_key = image_id
            if cache_key in self.reasoning_cache:
                logger.info("Retrieved reasoning from cache")
                return self.reasoning_cache[cache_key]

        # Execute reasoning based on mode
        if self.use_llm:
            logger.info("Executing LLM-based medical chain-of-thought reasoning")
            prompt = self._construct_medical_cot_prompt(vision_output, patient_metadata)
            generated_output = self._perform_llm_inference(prompt)

            # Parse the output to separate reasoning and explanation
            parsed_output = self._parse_model_output(generated_output)
            reasoning_text = parsed_output["reasoning"]
            explanation_text = parsed_output["explanation"]

            reasoning_mode = "llm_medical_cot"
        else:
            logger.info("Executing rule-based chain-of-thought reasoning")
            reasoning_steps = self._execute_rule_based_reasoning(
                vision_output, patient_metadata
            )

            # Format reasoning steps with separation
            formatted_output = self._format_reasoning_steps_for_separation(
                reasoning_steps
            )
            reasoning_text = formatted_output["reasoning"]
            explanation_text = formatted_output["explanation"]

            reasoning_mode = "rule_based"

        # Extract stage information
        stage_info = self.STAGE_DEFINITIONS.get(stage, self.STAGE_DEFINITIONS[0])

        clinical_recommendations = self._generate_clinical_recommendations(
            stage, confidence, patient_metadata
        )

        # Construct output with separated reasoning and explanation
        result = {
            "image_id": image_id,
            "reasoning": reasoning_text,  # Chain-of-thought reasoning steps
            "explanation": explanation_text,  # Final clinical explanation
            "reasoning_mode": reasoning_mode,
            "stage_name": stage_info["name"],
            "severity_level": stage_info["severity"],
            "clinical_recommendation": clinical_recommendations["recommendation"],
            "confidence_level": self._classify_confidence_level(confidence),
        }

        # Cache result if enabled
        if self.enable_caching and self.reasoning_cache is not None:
            cache_key = image_id
            self.reasoning_cache[cache_key] = result

        logger.info(
            f"Reasoning complete for image_id {image_id}: "
            f"{stage_info['name']} with {confidence:.1%} confidence"
        )

        return result

    def _classify_confidence_level(self, confidence: float) -> str:
        """
        Classify confidence score into categorical levels.

        Parameters
        ----------
        confidence : float
            Prediction confidence (0.0 to 1.0).

        Returns
        -------
        str
            Confidence level classification.
        """
        if confidence >= self.CONFIDENCE_THRESHOLDS["high"]:
            return "high"
        elif confidence >= self.CONFIDENCE_THRESHOLDS["moderate"]:
            return "moderate"
        else:
            return "low"

    def clear_cache(self):
        """Clear the reasoning cache to free memory."""
        if self.reasoning_cache is not None:
            self.reasoning_cache.clear()
            logger.info("Reasoning cache cleared")

    def switch_reasoning_mode(self, use_llm: bool):
        """
        Switch between LLM and rule-based reasoning modes.

        Parameters
        ----------
        use_llm : bool
            True for LLM mode, False for rule-based mode.
        """
        if use_llm and not self.use_llm:
            logger.warning(
                "Cannot switch to LLM mode: model not loaded during initialization"
            )
            return

        self.use_llm = use_llm
        logger.info(f"Reasoning mode switched to: {'LLM' if use_llm else 'rule-based'}")

    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """
        Retrieve statistics about the reasoning agent's operation.

        Returns
        -------
        dict
            Statistics including cache size, device info, and model status.
        """
        stats = {
            "reasoning_mode": "llm_medical_cot" if self.use_llm else "rule_based",
            "model_name": self.model_id,
            "device": str(self.device),
            "cache_enabled": self.enable_caching,
            "cache_size": len(self.reasoning_cache) if self.reasoning_cache else 0,
            "model_loaded": self.model is not None,
        }

        return stats
