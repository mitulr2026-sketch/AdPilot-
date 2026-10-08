# app/data_quality.py

from __future__ import annotations

import math
import re
from typing import Any


# ---------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _clean_number(value: Any) -> Any:
    """
    Convert common messy Excel/CSV values into numbers.

    Examples:
        "12,500" -> 12500
        " ₹50,000 " -> 50000
        "25%" -> 25
        "" -> None
        "N/A" -> None
    """

    if value is None:
        return None

    if _is_number(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    if not isinstance(value, str):
        return value

    value = value.strip()

    if not value:
        return None

    if value.lower() in {
        "n/a",
        "na",
        "none",
        "null",
        "-",
        "--",
        "unknown",
        "not available",
    }:
        return None

    cleaned = re.sub(r"[₹$€£,\s]", "", value)

    if cleaned.endswith("%"):
        try:
            return float(cleaned[:-1])
        except ValueError:
            return value

    try:
        number = float(cleaned)

        if math.isfinite(number):
            return number

    except ValueError:
        pass

    return value


def _clean_text(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, str):
        value = re.sub(r"\s+", " ", value).strip()
        return value if value else None

    return value


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    issues = []

    campaign = record.get("campaign", "unknown")

    # -----------------------------------------------------
    # Conversion rate
    # -----------------------------------------------------

    conversion = record.get("conversion_rate")

    if conversion is not None:

        if not _is_number(conversion):
            issues.append({
                "field": "conversion_rate",
                "severity": "error",
                "message": "Conversion rate is not numeric.",
                "value": conversion,
            })

        elif conversion < 0:
            issues.append({
                "field": "conversion_rate",
                "severity": "error",
                "message": "Conversion rate cannot be negative.",
                "value": conversion,
            })

        elif conversion > 100:
            issues.append({
                "field": "conversion_rate",
                "severity": "error",
                "message": "Conversion rate cannot exceed 100%.",
                "value": conversion,
            })

    # -----------------------------------------------------
    # CTR
    # -----------------------------------------------------

    ctr = record.get("ctr")

    if ctr is not None:

        if not _is_number(ctr):
            issues.append({
                "field": "ctr",
                "severity": "error",
                "message": "CTR is not numeric.",
                "value": ctr,
            })

        elif ctr < 0:
            issues.append({
                "field": "ctr",
                "severity": "error",
                "message": "CTR cannot be negative.",
                "value": ctr,
            })

        elif ctr > 100:
            issues.append({
                "field": "ctr",
                "severity": "error",
                "message": "CTR cannot exceed 100%.",
                "value": ctr,
            })

    # -----------------------------------------------------
    # Numeric fields
    # -----------------------------------------------------

    for field in (
        "budget",
        "spend",
        "revenue",
        "impressions",
        "clicks",
        "conversions",
        "inventory",
        "margin",
    ):
        value = record.get(field)

        if value is None:
            continue

        if not _is_number(value):
            issues.append({
                "field": field,
                "severity": "error",
                "message": f"{field} is not numeric.",
                "value": value,
            })
            continue

        if field in {
            "budget",
            "spend",
            "revenue",
            "impressions",
            "clicks",
            "conversions",
            "inventory",
        } and value < 0:

            issues.append({
                "field": field,
                "severity": "error",
                "message": f"{field} cannot be negative.",
                "value": value,
            })

    # -----------------------------------------------------
    # Whole-number metrics
    # -----------------------------------------------------

    for field in ("impressions", "clicks", "conversions", "inventory"):

        value = record.get(field)

        if _is_number(value) and not float(value).is_integer():

            issues.append({
                "field": field,
                "severity": "warning",
                "message": f"{field} should normally be a whole number.",
                "value": value,
            })

    # -----------------------------------------------------
    # Margin
    # -----------------------------------------------------

    margin = record.get("margin")

    if _is_number(margin):

        if margin < 0:
            issues.append({
                "field": "margin",
                "severity": "error",
                "message": "Margin cannot be negative.",
                "value": margin,
            })

        elif margin > 100:
            issues.append({
                "field": "margin",
                "severity": "error",
                "message": "Margin cannot exceed 100%.",
                "value": margin,
            })

    # -----------------------------------------------------
    # Logical relationships
    # -----------------------------------------------------

    impressions = record.get("impressions")
    clicks = record.get("clicks")
    conversions = record.get("conversions")

    if (
        _is_number(clicks)
        and _is_number(impressions)
        and clicks > impressions
    ):
        issues.append({
            "field": "clicks",
            "severity": "error",
            "message": "Clicks cannot be greater than impressions.",
            "value": clicks,
        })

    if (
        _is_number(conversions)
        and _is_number(clicks)
        and conversions > clicks
    ):
        issues.append({
            "field": "conversions",
            "severity": "error",
            "message": "Conversions cannot be greater than clicks.",
            "value": conversions,
        })

    # -----------------------------------------------------
    # Missing campaign
    # -----------------------------------------------------

    if not campaign or str(campaign).strip() == "":
        issues.append({
            "field": "campaign",
            "severity": "warning",
            "message": "Campaign name is missing.",
            "value": campaign,
        })

    return issues


# ---------------------------------------------------------
# Automatic correction
# ---------------------------------------------------------

def auto_correct_record(record: dict[str, Any]):
    """
    Safely correct values that can be corrected without guessing.

    Returns:
        corrected_record, corrections
    """

    record = dict(record)
    corrections = []

    # -----------------------------------------------------
    # Numeric conversion
    # -----------------------------------------------------

    numeric_fields = {
        "budget",
        "spend",
        "revenue",
        "conversion_rate",
        "ctr",
        "impressions",
        "clicks",
        "conversions",
        "inventory",
        "margin",
    }

    for field in numeric_fields:

        if field not in record:
            continue

        original = record[field]

        cleaned = _clean_number(original)

        if cleaned != original:

            record[field] = cleaned

            corrections.append({
                "field": field,
                "type": "format_correction",
                "old_value": original,
                "new_value": cleaned,
                "message": f"Cleaned {field}.",
            })

    # -----------------------------------------------------
    # Text cleanup
    # -----------------------------------------------------

    for field, value in record.items():

        if isinstance(value, str):

            cleaned = _clean_text(value)

            if cleaned != value:

                record[field] = cleaned

                corrections.append({
                    "field": field,
                    "type": "text_cleanup",
                    "old_value": value,
                    "new_value": cleaned,
                    "message": f"Cleaned text in {field}.",
                })

    # -----------------------------------------------------
    # Recalculate CTR
    #
    # Only when impressions and clicks are valid.
    # -----------------------------------------------------

    impressions = record.get("impressions")
    clicks = record.get("clicks")

    if (
        _is_number(impressions)
        and _is_number(clicks)
        and impressions > 0
        and 0 <= clicks <= impressions
    ):

        calculated_ctr = round((clicks / impressions) * 100, 4)

        existing_ctr = record.get("ctr")

        if existing_ctr is None or (
            _is_number(existing_ctr)
            and abs(existing_ctr - calculated_ctr) > 0.01
        ):

            record["ctr"] = calculated_ctr

            corrections.append({
                "field": "ctr",
                "type": "calculation_correction",
                "old_value": existing_ctr,
                "new_value": calculated_ctr,
                "message": "CTR recalculated from clicks and impressions.",
            })

    # -----------------------------------------------------
    # Recalculate conversion rate
    # -----------------------------------------------------

    clicks = record.get("clicks")
    conversions = record.get("conversions")

    if (
        _is_number(clicks)
        and _is_number(conversions)
        and clicks > 0
        and 0 <= conversions <= clicks
    ):

        calculated_conversion_rate = round(
            (conversions / clicks) * 100,
            4
        )

        existing_conversion = record.get("conversion_rate")

        if existing_conversion is None or (
            _is_number(existing_conversion)
            and abs(
                existing_conversion - calculated_conversion_rate
            ) > 0.01
        ):

            record["conversion_rate"] = calculated_conversion_rate

            corrections.append({
                "field": "conversion_rate",
                "type": "calculation_correction",
                "old_value": existing_conversion,
                "new_value": calculated_conversion_rate,
                "message": (
                    "Conversion rate recalculated "
                    "from conversions and clicks."
                ),
            })

    return record, corrections


# ---------------------------------------------------------
# Main cleaning + validation function
# ---------------------------------------------------------

def clean_and_validate(
    records: list[dict[str, Any]]
) -> dict[str, Any]:

    cleaned_records = []
    all_issues = []
    all_corrections = []

    seen_campaigns = set()

    for index, original in enumerate(records):

        # -------------------------------------------------
        # Copy original record
        # -------------------------------------------------

        record = dict(original)

        # -------------------------------------------------
        # Automatic correction
        # -------------------------------------------------

        record, corrections = auto_correct_record(record)

        for correction in corrections:
            correction["row"] = index + 1
            correction["campaign"] = record.get("campaign")

        all_corrections.extend(corrections)

        # -------------------------------------------------
        # Validation AFTER correction
        # -------------------------------------------------

        issues = validate_record(record)

        for issue in issues:
            issue["row"] = index + 1
            issue["campaign"] = record.get("campaign")

        # -------------------------------------------------
        # Duplicate campaign detection
        # -------------------------------------------------

        campaign = record.get("campaign")

        if campaign:

            campaign_key = str(campaign).strip().lower()

            if campaign_key in seen_campaigns:

                issues.append({
                    "row": index + 1,
                    "campaign": campaign,
                    "field": "campaign",
                    "severity": "warning",
                    "message": "Duplicate campaign detected.",
                    "value": campaign,
                })

            seen_campaigns.add(campaign_key)

        # -------------------------------------------------
        # Store
        # -------------------------------------------------

        all_issues.extend(issues)
        cleaned_records.append(record)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    errors = [
        x for x in all_issues
        if x["severity"] == "error"
    ]

    warnings = [
        x for x in all_issues
        if x["severity"] == "warning"
    ]

    return {
        "data": cleaned_records,

        "validation": {
            "valid": len(errors) == 0,
            "total_records": len(cleaned_records),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "correction_count": len(all_corrections),
            "issues": all_issues,
            "corrections": all_corrections,
        },
    }