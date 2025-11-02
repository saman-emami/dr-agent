"""
governor_agent.py

A self-contained class for policy validation, logical consistency checks,
and audit logging. Enforces governance rules and provides audit-ready outputs.
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json

# Configure logger for module
logger = logging.getLogger("GovernorAgent")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[GovernorAgent] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ValidationRule:
    """
    Structured representation of a governance validation rule.

    Attributes
    ----------
    rule_name : str
        Unique identifier for the rule.
    rule_type : str
        Category of validation (e.g., 'consistency', 'threshold', 'policy').
    passed : bool
        Whether the rule validation succeeded.
    message : str
        Human-readable explanation of the validation result.
    severity : str
        Impact level: 'critical', 'warning', or 'info'.
    """

    rule_name: str
    rule_type: str
    passed: bool
    message: str
    severity: str


@dataclass
class AuditRecord:
    """
    Complete audit trail for a single analysis request.

    Attributes
    ----------
    trace_id : str
        Unique identifier for this analysis session.
    timestamp : str
        ISO 8601 formatted timestamp.
    image_id : str
        Identifier of the analyzed retinal image.
    prediction : str
        Final DR stage prediction.
    confidence : float
        Model confidence score.
    validated : bool
        Whether all governance checks passed.
    validation_rules : list of dict
        Results from all validation rules applied.
    model_version : str
        Version identifier for deployed models.
    reasoning_summary : str
        Condensed explanation from ReasonerAgent.
    flags : list of str
        Any warnings or special conditions detected.
    """

    trace_id: str
    timestamp: str
    image_id: str
    prediction: str
    confidence: float
    validated: bool
    validation_rules: List[Dict[str, Any]]
    model_version: str
    reasoning_summary: str
    flags: List[str]


class GovernorAgent:
    """
    GovernorAgent for policy enforcement and audit logging in DR detection system.

    Validates logical consistency between vision and reasoning outputs, enforces
    clinical safety thresholds, and maintains comprehensive audit trails.

    Parameters
    ----------
    model_version : str, optional
        Version identifier for the deployed model ensemble (default: "v1.0").
    enable_strict_mode : bool, optional
        If True, fail on any validation warning; if False, only fail on critical
        errors (default: False).
    confidence_threshold : float, optional
        Minimum acceptable confidence score for high-severity predictions (default: 0.65).
    max_audit_records : int, optional
        Maximum number of audit records to retain in memory (default: 200).
    """

    # Stage definitions for validation
    STAGE_DEFINITIONS = {
        0: {"name": "No DR", "severity": "none", "urgency": "routine"},
        1: {"name": "Mild DR", "severity": "minimal", "urgency": "routine"},
        2: {"name": "Moderate DR", "severity": "moderate", "urgency": "semi-urgent"},
        3: {"name": "Severe DR", "severity": "severe", "urgency": "urgent"},
        4: {"name": "Proliferative DR", "severity": "critical", "urgency": "emergent"},
    }

    # Validation thresholds
    CONFIDENCE_THRESHOLDS = {
        "minimum_acceptable": 0.50,
        "high_severity_minimum": 0.65,
        "critical_severity_minimum": 0.70,
    }

    def __init__(
        self,
        model_version: str = "v1.0",
        enable_strict_mode: bool = False,
        confidence_threshold: float = 0.65,
        max_audit_records: int = 200,
    ):
        self.model_version = model_version
        self.enable_strict_mode = enable_strict_mode
        self.confidence_threshold = confidence_threshold
        self.max_audit_records = max_audit_records

        # In-memory audit trail storage
        self.audit_records: List[AuditRecord] = []

        logger.info(f"GovernorAgent initialized with model version: {model_version}")
        logger.info(
            f"Strict mode: {enable_strict_mode}, Confidence threshold: {confidence_threshold}"
        )

    def _validate_stage_confidence_alignment(
        self,
        stage: int,
        confidence: float,
    ) -> ValidationRule:
        """
        Verify that confidence score meets requirements for predicted severity.

        Critical and severe stages require higher confidence to minimize false positives
        that could lead to unnecessary invasive procedures.

        Parameters
        ----------
        stage : int
            Predicted DR stage (0-4).
        confidence : float
            Model confidence score.

        Returns
        -------
        ValidationRule
            Validation result with pass/fail status.
        """
        stage_info = self.STAGE_DEFINITIONS.get(stage, self.STAGE_DEFINITIONS[0])
        severity = stage_info["severity"]

        if severity in ["critical", "severe"]:
            required_confidence = self.CONFIDENCE_THRESHOLDS[
                "critical_severity_minimum"
            ]
            if confidence < required_confidence:
                return ValidationRule(
                    rule_name="stage_confidence_alignment",
                    rule_type="threshold",
                    passed=False,
                    message=(
                        f"Confidence {confidence:.2%} below required threshold "
                        f"{required_confidence:.2%} for {severity} severity prediction"
                    ),
                    severity="critical",
                )

        elif severity == "moderate":
            required_confidence = self.CONFIDENCE_THRESHOLDS["high_severity_minimum"]
            if confidence < required_confidence:
                return ValidationRule(
                    rule_name="stage_confidence_alignment",
                    rule_type="threshold",
                    passed=False,
                    message=(
                        f"Confidence {confidence:.2%} below recommended threshold "
                        f"{required_confidence:.2%} for moderate severity"
                    ),
                    severity="warning",
                )

        # Check minimum acceptable confidence for any prediction
        minimum_confidence = self.CONFIDENCE_THRESHOLDS["minimum_acceptable"]
        if confidence < minimum_confidence:
            return ValidationRule(
                rule_name="stage_confidence_alignment",
                rule_type="threshold",
                passed=False,
                message=f"Confidence {confidence:.2%} below minimum threshold {minimum_confidence:.2%}",
                severity="critical",
            )

        return ValidationRule(
            rule_name="stage_confidence_alignment",
            rule_type="threshold",
            passed=True,
            message=f"Confidence {confidence:.2%} meets requirements for {severity} severity",
            severity="info",
        )

    def _validate_vision_reasoning_consistency(
        self,
        vision_output: Dict[str, Any],
        reasoning_output: Dict[str, Any],
    ) -> ValidationRule:
        """
        Ensure vision and reasoning agents produced consistent interpretations.

        Parameters
        ----------
        vision_output : dict
            Output from VisionAgent containing stage and confidence.
        reasoning_output : dict
            Output from ReasonerAgent containing stage_name and severity_level.

        Returns
        -------
        ValidationRule
            Validation result checking for consistency.
        """
        vision_stage = vision_output.get("stage", -1)
        vision_stage_name = self.STAGE_DEFINITIONS.get(vision_stage, {}).get(
            "name", "Unknown"
        )

        reasoning_stage_name = reasoning_output.get("stage_name", "")
        reasoning_severity = reasoning_output.get("severity_level", "")

        expected_severity = self.STAGE_DEFINITIONS.get(vision_stage, {}).get(
            "severity", ""
        )

        # Check stage name consistency
        if vision_stage_name != reasoning_stage_name:
            return ValidationRule(
                rule_name="vision_reasoning_consistency",
                rule_type="consistency",
                passed=False,
                message=(
                    f"Stage mismatch: Vision predicted '{vision_stage_name}' but "
                    f"Reasoner interpreted as '{reasoning_stage_name}'"
                ),
                severity="critical",
            )

        # Check severity level consistency
        if expected_severity != reasoning_severity:
            return ValidationRule(
                rule_name="vision_reasoning_consistency",
                rule_type="consistency",
                passed=False,
                message=(
                    f"Severity mismatch: Expected '{expected_severity}' but "
                    f"Reasoner reported '{reasoning_severity}'"
                ),
                severity="warning",
            )

        return ValidationRule(
            rule_name="vision_reasoning_consistency",
            rule_type="consistency",
            passed=True,
            message="Vision and reasoning outputs are consistent",
            severity="info",
        )

    def _validate_stage_range(self, stage: int) -> ValidationRule:
        """
        Verify that predicted stage falls within valid range.

        Parameters
        ----------
        stage : int
            Predicted DR stage.

        Returns
        -------
        ValidationRule
            Validation result for stage range check.
        """
        if stage < 0 or stage > 4:
            return ValidationRule(
                rule_name="stage_range_validation",
                rule_type="policy",
                passed=False,
                message=f"Invalid stage {stage}: must be between 0 and 4",
                severity="critical",
            )

        return ValidationRule(
            rule_name="stage_range_validation",
            rule_type="policy",
            passed=True,
            message=f"Stage {stage} is within valid range",
            severity="info",
        )

    def _validate_required_fields(
        self,
        vision_output: Dict[str, Any],
        reasoning_output: Dict[str, Any],
    ) -> ValidationRule:
        """
        Ensure all required fields are present in agent outputs.

        Parameters
        ----------
        vision_output : dict
            Output from VisionAgent.
        reasoning_output : dict
            Output from ReasonerAgent.

        Returns
        -------
        ValidationRule
            Validation result for required fields check.
        """
        required_vision_fields = ["image_id", "stage", "confidence", "key_regions"]
        required_reasoning_fields = [
            "image_id",
            "explanation",
            "stage_name",
            "clinical_recommendation",
        ]

        missing_fields = []

        for field in required_vision_fields:
            if field not in vision_output:
                missing_fields.append(f"vision_output.{field}")

        for field in required_reasoning_fields:
            if field not in reasoning_output:
                missing_fields.append(f"reasoning_output.{field}")

        if missing_fields:
            return ValidationRule(
                rule_name="required_fields_validation",
                rule_type="policy",
                passed=False,
                message=f"Missing required fields: {', '.join(missing_fields)}",
                severity="critical",
            )

        return ValidationRule(
            rule_name="required_fields_validation",
            rule_type="policy",
            passed=True,
            message="All required fields present",
            severity="info",
        )

    def _check_special_conditions(
        self,
        stage: int,
        confidence: float,
        reasoning_output: Dict[str, Any],
    ) -> List[str]:
        """
        Identify special conditions or flags that require attention.

        Parameters
        ----------
        stage : int
            Predicted DR stage.
        confidence : float
            Model confidence score.
        reasoning_output : dict
            Output from ReasonerAgent.

        Returns
        -------
        list of str
            List of flag messages for special conditions.
        """
        flags = []

        # Flag low confidence predictions
        if confidence < self.CONFIDENCE_THRESHOLDS["high_severity_minimum"]:
            flags.append(
                f"LOW_CONFIDENCE: {confidence:.2%} - manual review recommended"
            )

        # Flag high severity cases for urgent review
        stage_info = self.STAGE_DEFINITIONS.get(stage, {})
        if stage_info.get("urgency") in ["urgent", "emergent"]:
            flags.append(
                f"HIGH_URGENCY: {stage_info.get('urgency')} - expedited clinical review required"
            )

        # Flag borderline confidence for critical stages
        if stage >= 3 and confidence < 0.80:
            flags.append(
                "BORDERLINE_CONFIDENCE: Critical stage with moderate confidence - confirm diagnosis"
            )

        # Check reasoning confidence level
        reasoning_confidence_level = reasoning_output.get("confidence_level", "")
        if reasoning_confidence_level == "low":
            flags.append(
                "REASONER_UNCERTAINTY: Reasoning agent flagged uncertainty in interpretation"
            )

        return flags

    def _generate_trace_id(self) -> str:
        """
        Generate unique trace identifier for audit trail.

        Returns
        -------
        str
            UUID4 trace identifier.
        """
        return str(uuid.uuid4())

    def _get_current_timestamp(self) -> str:
        """
        Generate ISO 8601 formatted timestamp.

        Returns
        -------
        str
            Current timestamp in ISO format.
        """
        return datetime.utcnow().isoformat() + "Z"

    def _create_audit_record(
        self,
        trace_id: str,
        vision_output: Dict[str, Any],
        reasoning_output: Dict[str, Any],
        validation_rules: List[ValidationRule],
        validated: bool,
        flags: List[str],
    ) -> AuditRecord:
        """
        Construct complete audit record for the analysis.

        Parameters
        ----------
        trace_id : str
            Unique identifier for this analysis.
        vision_output : dict
            Output from VisionAgent.
        reasoning_output : dict
            Output from ReasonerAgent.
        validation_rules : list of ValidationRule
            All validation rules that were applied.
        validated : bool
            Overall validation status.
        flags : list of str
            Special condition flags.

        Returns
        -------
        AuditRecord
            Complete audit record.
        """
        stage = vision_output.get("stage", -1)
        stage_name = self.STAGE_DEFINITIONS.get(stage, {}).get("name", "Unknown")

        # Create condensed reasoning summary
        reasoning_summary = reasoning_output.get(
            "clinical_recommendation", "No recommendation available"
        )
        if len(reasoning_summary) > 200:
            reasoning_summary = reasoning_summary[:197] + "..."

        audit_record = AuditRecord(
            trace_id=trace_id,
            timestamp=self._get_current_timestamp(),
            image_id=vision_output.get("image_id", "unknown"),
            prediction=stage_name,
            confidence=vision_output.get("confidence", 0.0),
            validated=validated,
            validation_rules=[asdict(rule) for rule in validation_rules],
            model_version=self.model_version,
            reasoning_summary=reasoning_summary,
            flags=flags,
        )

        return audit_record

    def _store_audit_record(self, audit_record: AuditRecord):
        """
        Store audit record in memory with size limit enforcement.

        Parameters
        ----------
        audit_record : AuditRecord
            Record to store.
        """
        self.audit_records.append(audit_record)

        # Enforce maximum storage limit
        if len(self.audit_records) > self.max_audit_records:
            removed_count = len(self.audit_records) - self.max_audit_records
            self.audit_records = self.audit_records[-self.max_audit_records :]
            logger.warning(
                f"Audit storage limit reached. Removed {removed_count} oldest records."
            )

    def govern(
        self,
        vision_output: Dict[str, Any],
        reasoning_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply governance policies and generate audit-ready response.

        Validates consistency, enforces thresholds, identifies special conditions,
        and creates comprehensive audit trail.

        Parameters
        ----------
        vision_output : dict
            Output from VisionAgent containing:
            - image_id: str
            - stage: int
            - confidence: float
            - key_regions: numpy.ndarray
        reasoning_output : dict
            Output from ReasonerAgent containing:
            - image_id: str
            - explanation: str
            - reasoning': str
            - stage_name: str
            - severity_level: str
            - clinical_recommendation: str
            - confidence_level: str

        Returns
        -------
        dict
            {
                'image_id': str,
                'prediction': str,
                'confidence': float,
                'explanation': str,
                'reasoning': str,
                'governance': {
                    'validated': bool,
                    'timestamp': str,
                    'model_version': str,
                    'trace_id': str,
                    'validation_summary': str,
                    'flags': list of str
                }
            }

        Raises
        ------
        ValueError
            If required fields are missing or validation fails in strict mode.
        """
        trace_id = self._generate_trace_id()
        logger.info(f"Starting governance validation for trace_id: {trace_id}")

        # Apply all validation rules
        validation_rules = []

        # Rule 1: Validate required fields
        required_fields_rule = self._validate_required_fields(
            vision_output, reasoning_output
        )

        validation_rules.append(required_fields_rule)
        if not required_fields_rule.passed:
            logger.error(
                f"Required fields validation failed: {required_fields_rule.message}"
            )
            raise ValueError(required_fields_rule.message)

        # Rule 2: Validate stage range
        stage = vision_output.get("stage", -1)
        stage_range_rule = self._validate_stage_range(stage)
        validation_rules.append(stage_range_rule)

        # Rule 3: Validate confidence alignment with severity
        confidence = vision_output.get("confidence", 0.0)
        confidence_rule = self._validate_stage_confidence_alignment(stage, confidence)
        validation_rules.append(confidence_rule)

        # Rule 4: Validate consistency between vision and reasoning
        consistency_rule = self._validate_vision_reasoning_consistency(
            vision_output, reasoning_output
        )
        validation_rules.append(consistency_rule)

        # Check for special conditions
        flags = self._check_special_conditions(stage, confidence, reasoning_output)

        # Determine overall validation status
        critical_failures = [
            rule
            for rule in validation_rules
            if not rule.passed and rule.severity == "critical"
        ]
        warning_failures = [
            rule
            for rule in validation_rules
            if not rule.passed and rule.severity == "warning"
        ]

        # In strict mode, warnings also cause validation failure
        if self.enable_strict_mode:
            validated = len(critical_failures) == 0 and len(warning_failures) == 0
        else:
            validated = len(critical_failures) == 0

        # Log validation results
        if critical_failures:
            for rule in critical_failures:
                logger.error(f"CRITICAL: {rule.rule_name} - {rule.message}")
        if warning_failures:
            for rule in warning_failures:
                logger.warning(f"WARNING: {rule.rule_name} - {rule.message}")

        # Create and store audit record
        audit_record = self._create_audit_record(
            trace_id=trace_id,
            vision_output=vision_output,
            reasoning_output=reasoning_output,
            validation_rules=validation_rules,
            validated=validated,
            flags=flags,
        )
        self._store_audit_record(audit_record)

        # Generate validation summary
        passed_rules = sum(1 for rule in validation_rules if rule.passed)
        total_rules = len(validation_rules)
        validation_summary = f"{passed_rules}/{total_rules} rules passed"
        if critical_failures:
            validation_summary += f", {len(critical_failures)} critical failures"
        if warning_failures:
            validation_summary += f", {len(warning_failures)} warnings"

        logger.info(
            f"Governance complete for trace_id {trace_id}: "
            f"validated={validated}, {validation_summary}"
        )

        # Construct governed response
        stage_info = self.STAGE_DEFINITIONS.get(stage, self.STAGE_DEFINITIONS[0])

        response = {
            "image_id": vision_output.get("image_id", "unknown"),
            "key_regions": vision_output.get("key_regions", "unknown"),
            "prediction": stage_info["name"],
            "confidence": confidence,
            "explanation": reasoning_output.get(
                "explanation", "No explanation available"
            ),
            "reasoning": reasoning_output.get("reasoning", "No reasoning available"),
            "governance": {
                "validated": validated,
                "timestamp": audit_record.timestamp,
                "model_version": self.model_version,
                "trace_id": trace_id,
                "validation_summary": validation_summary,
                "flags": flags,
            },
        }

        return response

    def get_audit_logs(
        self,
        limit: Optional[int] = None,
        filter_validated: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit logs with optional filtering.

        Parameters
        ----------
        limit : int, optional
            Maximum number of records to return (most recent first).
        filter_validated : bool, optional
            If provided, only return records matching this validation status.

        Returns
        -------
        list of dict
            Audit records as dictionaries.
        """
        records = self.audit_records.copy()

        # Apply validation filter
        if filter_validated is not None:
            records = [r for r in records if r.validated == filter_validated]

        # Sort by timestamp descending (most recent first)
        records.sort(key=lambda r: r.timestamp, reverse=True)

        # Apply limit
        if limit is not None:
            records = records[:limit]

        logger.info(f"Retrieved {len(records)} audit records")
        return [asdict(record) for record in records]

    def get_governance_statistics(self) -> Dict[str, Any]:
        """
        Generate statistics about governance operations.

        Returns
        -------
        dict
            Statistics including validation rates, common flags, and performance metrics.
        """
        if not self.audit_records:
            return {
                "total_analyses": 0,
                "validation_rate": 0.0,
                "most_common_flags": [],
                "model_version": self.model_version,
            }

        total_analyses = len(self.audit_records)
        validated_count = sum(1 for record in self.audit_records if record.validated)
        validation_rate = (
            validated_count / total_analyses if total_analyses > 0 else 0.0
        )

        # Collect all flags
        all_flags = []
        for record in self.audit_records:
            all_flags.extend(record.flags)

        # Count flag occurrences
        flag_counts = {}
        for flag in all_flags:
            flag_type = flag.split(":")[0]
            flag_counts[flag_type] = flag_counts.get(flag_type, 0) + 1

        # Sort flags by frequency
        most_common_flags = sorted(
            flag_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        statistics = {
            "total_analyses": total_analyses,
            "validated_count": validated_count,
            "failed_count": total_analyses - validated_count,
            "validation_rate": round(validation_rate, 4),
            "most_common_flags": most_common_flags,
            "model_version": self.model_version,
            "strict_mode_enabled": self.enable_strict_mode,
            "storage_utilization": f"{len(self.audit_records)}/{self.max_audit_records}",
        }

        logger.info(
            f"Generated governance statistics: {validation_rate:.2%} validation rate"
        )
        return statistics

    def clear_audit_logs(self):
        """Clear all stored audit records."""
        record_count = len(self.audit_records)
        self.audit_records.clear()
        logger.info(f"Cleared {record_count} audit records")

    def export_audit_logs(self, filepath: str, format_type: str = "json"):
        """
        Export audit logs to file.

        Parameters
        ----------
        filepath : str
            Output file path.
        format_type : str, optional
            Export format: 'json' or 'jsonl' (default: 'json').
        """
        records = [asdict(record) for record in self.audit_records]

        if format_type == "json":
            with open(filepath, "w") as file:
                json.dump(records, file, indent=2)
        elif format_type == "jsonl":
            with open(filepath, "w") as file:
                for record in records:
                    file.write(json.dumps(record) + "\n")
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        logger.info(f"Exported {len(records)} audit records to {filepath}")
