"""
Pydantic schema models for validating input datasets used by the accuracy_rag test container.

This module provides validation for:
1. JSONL files containing RAG accuracy test data
2. CSV files containing instruction-following test data

These schemas ensure input files have the required columns and data structure
before processing begins.
"""

from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field, ValidationError, validator

from asqi.datasets import Dataset


class RAGAccuracyRow(BaseModel):
    """Schema for a single row in the RAG accuracy JSONL dataset."""

    # Primary question field (required)
    question: str = Field(
        ..., description="The question to ask the RAG system", min_length=1
    )

    # Answer field (required)
    answer: str = Field(
        ..., description="The ground truth answer for evaluation", min_length=1
    )

    # Context/retrieval field (required, can be string or list)
    context: Union[List[str], str] = Field(
        ..., description="Retrieved context documents (can be single string or list)"
    )

    @validator("context", pre=True)
    def normalize_context(cls, v):
        """Normalize context field to always be a list of strings."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(ctx) for ctx in v if ctx]
        return []


class InstructionFollowingRow(BaseModel):
    """Schema for a single row in the instruction-following CSV dataset."""

    # Transformed question (required)
    transformed_question: str = Field(
        ...,
        description="The instruction-augmented question to ask the system",
        min_length=1,
    )

    # Original question (required)
    original_question: str = Field(
        ...,
        description="The original question before instruction augmentation",
        min_length=1,
    )

    # Instruction type (required)
    instruction_type: str = Field(
        ..., description="Type/category of instruction applied", min_length=1
    )

    # Instruction metadata (required, JSON string)
    instruction_type_metadata: Union[str, Dict[str, Any]] = Field(
        ...,
        description="Additional metadata about the instruction (JSON string or dict)",
    )

    @validator("instruction_type_metadata", pre=True)
    def parse_metadata(cls, v):
        """Parse instruction_type_metadata from JSON string to dict if needed."""
        if v is None:
            return {}
        if isinstance(v, str):
            try:
                import json

                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return {"raw": v}
        if isinstance(v, dict):
            return v
        return {"raw": str(v)}


class DatasetValidator:
    """Validator class for checking entire datasets."""

    @staticmethod
    def validate_rag_accuracy_hf(
        dataset: Dataset, max_errors: int = 10
    ) -> Dict[str, Any]:
        """Validate a RAG accuracy dataset from a Hugging Face Dataset object.

        Args:
            dataset: Hugging Face Dataset object
            max_errors: Maximum number of validation errors to collect
        Returns:
            Dict with validation results:
            {
                "valid": bool,
                "total_rows": int,
                "valid_rows": int,
                "errors": List[str]
            }
        """
        errors = []
        valid_rows = 0
        total_rows = 0

        for row_num, raw_data in enumerate(dataset):
            total_rows += 1

            try:
                RAGAccuracyRow(**raw_data)
                valid_rows += 1
            except ValidationError as e:
                if len(errors) < max_errors:
                    errors.append(f"Row {row_num}: {e}")
            except Exception as e:
                if len(errors) < max_errors:
                    errors.append(f"Row {row_num}: Failed to parse - {e}")

        return {
            "valid": len(errors) == 0,
            "total_rows": total_rows,
            "valid_rows": valid_rows,
            "errors": errors,
        }

    @staticmethod
    def validate_rag_accuracy_jsonl(
        jsonl_path: str, max_errors: int = 10
    ) -> Dict[str, Any]:
        """Validate a RAG accuracy JSONL file.

        Args:
            jsonl_path: Path to the JSONL file
            max_errors: Maximum number of validation errors to collect

        Returns:
            Dict with validation results:
            {
                "valid": bool,
                "total_rows": int,
                "valid_rows": int,
                "errors": List[str]
            }
        """
        import json

        errors = []
        valid_rows = 0
        total_rows = 0

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    total_rows += 1

                    try:
                        raw_data = json.loads(line)
                        RAGAccuracyRow(**raw_data)
                        valid_rows += 1
                    except ValidationError as e:
                        if len(errors) < max_errors:
                            errors.append(f"Row {line_num}: {e}")
                    except Exception as e:
                        if len(errors) < max_errors:
                            errors.append(f"Row {line_num}: Failed to parse JSON - {e}")

        except FileNotFoundError:
            return {
                "valid": False,
                "total_rows": 0,
                "valid_rows": 0,
                "errors": [f"File not found: {jsonl_path}"],
            }
        except Exception as e:
            return {
                "valid": False,
                "total_rows": 0,
                "valid_rows": 0,
                "errors": [f"Failed to read file: {e}"],
            }

        return {
            "valid": len(errors) == 0,
            "total_rows": total_rows,
            "valid_rows": valid_rows,
            "errors": errors,
        }

    @staticmethod
    def validate_instruction_following_csv(
        csv_path: str, max_errors: int = 10
    ) -> Dict[str, Any]:
        """Validate an instruction-following CSV file.

        Args:
            csv_path: Path to the CSV file
            max_errors: Maximum number of validation errors to collect

        Returns:
            Dict with validation results:
            {
                "valid": bool,
                "total_rows": int,
                "valid_rows": int,
                "errors": List[str]
            }
        """
        import csv

        errors = []
        valid_rows = 0
        total_rows = 0

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row_num, raw_data in enumerate(
                    reader, 2
                ):  # Start at 2 (header is row 1)
                    total_rows += 1

                    try:
                        InstructionFollowingRow(**raw_data)
                        valid_rows += 1
                    except ValidationError as e:
                        if len(errors) < max_errors:
                            errors.append(f"Row {row_num}: {e}")
                    except Exception as e:
                        if len(errors) < max_errors:
                            errors.append(f"Row {row_num}: Failed to parse - {e}")

        except FileNotFoundError:
            return {
                "valid": False,
                "total_rows": 0,
                "valid_rows": 0,
                "errors": [f"File not found: {csv_path}"],
            }
        except Exception as e:
            return {
                "valid": False,
                "total_rows": 0,
                "valid_rows": 0,
                "errors": [f"Failed to read file: {e}"],
            }

        return {
            "valid": len(errors) == 0,
            "total_rows": total_rows,
            "valid_rows": valid_rows,
            "errors": errors,
        }


# Convenience functions for quick validation
def validate_rag_dataset(jsonl_path: str) -> bool:
    """Quick validation - returns True if dataset is valid."""
    result = DatasetValidator.validate_rag_accuracy_jsonl(jsonl_path)
    return result["valid"]


def validate_if_dataset(csv_path: str) -> bool:
    """Quick validation - returns True if dataset is valid."""
    result = DatasetValidator.validate_instruction_following_csv(csv_path)
    return result["valid"]
