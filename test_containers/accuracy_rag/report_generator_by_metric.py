import arakawa as ar
import json
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

# Chart color - neutral for all metrics
DEFAULT_CHART_COLOR = "#6c757d"  # Gray

# Special fields across all datasets that should NOT be rendered generically
# Only the essential ones are hidden and handled explicitly below
SPECIAL_FIELDS = {"question", "sut_answer", "metrics"}

# Remove fields you don't want to display in report, across all datasets
EXCLUDE_FIELDS = {"question_index", "instruction_type_metadata"}


# Comprehensive metric descriptions organized by layer
# Layer 1: Sample-Level Metrics (measured directly per sample, some are derived from others)
# Layer 2: System-Level Derived Metrics (aggregated across all samples)

# Layer 1: Sample-level metric descriptions
LAYER1_METRIC_DESCRIPTIONS: Dict[str, str] = {
    "answer_correctness": "Measures factual correctness against the reference answer (recall mode).",
    "faithfulness": "Assesses whether response claims are supported by retrieved contexts.",
    "helpfulness": "Measures relevance and completeness of the response to the user question.",
    "retrieval_correctness": "Combined precision/recall score evaluating retrieved contexts against references.",
    "hit@k": "Indicates whether the reference context appears within the top-k retrieved contexts.",
    "retrieval_hard_gate": "(Derived) Binary gate: passed if reference was found within top-k retrieved contexts. Pass (1) if Hit@K == 1, else Fail (0).",
    "faithfulness_hard_gate": "(Derived) Binary gate: passed when faithfulness score ≥ threshold (≥0.70). Pass (1) if Faithfulness ≥ 0.70, else Fail (0).",
    "hard_gate_pass": "(Derived) Binary indicator: 1.0 if all hard gates passed for the sample, else 0.0. Hard Gate Pass = Faithfulness Hard Gate × Retrieval Hard Gate; both gates must pass (=1) for this to be 1.",
    "conditional_task_success": "(Derived) Weighted quality score: 0.70×Answer Correctness + 0.30×Helpfulness when Hard Gate Pass is 1.0; 0 otherwise.",
    "instruction_following": "Score for how well the model followed explicit instructions/formatting.",
}

# Layer 2: System-level metric descriptions
LAYER2_METRIC_DESCRIPTIONS: Dict[str, str] = {
    "answer_correctness": "Average Answer Correctness (across all samples) — Factual accuracy on average.",
    "faithfulness": "Average Faithfulness (across all samples) — Support of claims by context on average.",
    "helpfulness": "Average Helpfulness (across all samples) — Relevance and completeness on average.",
    "retrieval_correctness": "Average Retrieval Correctness (across all samples) — Combined precision/recall of retrieval on average.",
    "hit@k": "Average Hit@K (across all samples) — Frequency of reference context appearing in top-k results on average.",
    "retrieval_gate_pass_rate": "Fraction of samples where retrieval hard gate was passed.",
    "faithfulness_gate_pass_rate": "Fraction of samples where faithfulness hard gate was passed (Faithfulness ≥ 0.70).",
    "hard_gate_pass_rate": "Fraction of samples passing both retrieval and faithfulness quality gates.",
    "conditional_task_success": "Average of Conditional Task Success — Quality of responses when gates pass.",
    "instruction_following": "Average Instruction Following (across all samples) — Consistency with formatting and instruction requirements.",
}

# Config: define which metrics are binary vs continuous. Can be extended as needed.
# If a metric is not listed here, we'll auto-detect binary metrics when all values are 0/1.
BINARY_METRICS = {
    "hard_gate_pass",
    "retrieval_hard_gate",
    "faithfulness_hard_gate",
}

CONTINUOUS_METRICS = {
    "instruction_following",
    "faithfulness",
    "answer_correctness",
    "helpfulness",
    "hit@k",
    "task_success",
    "retrieval_correctness",
}

# ============================================================================
# RECORD DETAIL COMPONENTS
# ============================================================================


def render_field_value(field_name: str, field_value: Any) -> str:
    """Dynamically render any field value based on its type"""
    formatted_name = field_name.replace("_", " ").title()

    # Handle lists (like contexts)
    if isinstance(field_value, list):
        if not field_value:
            return ""

        # Render full numbered list for any list-valued field
        html = f"<div style='margin-bottom: 1rem;'><h4 style='color: #667eea;'>🔖 {formatted_name}</h4>"
        html += "<ol style='margin: 0.5rem 0 0 1.25rem; color: #6c757d;'>"
        for i, item in enumerate(field_value, 1):
            item_str = str(item)
            # Truncate very long items for readability but keep most content
            truncated = item_str[:1000] + "..." if len(item_str) > 1000 else item_str
            html += f"<li style='margin-bottom: 0.5rem;'><div style='background: #f8f9fa; padding: 0.6rem; border-radius: 4px; font-size: 0.95em;'>{truncated}</div></li>"
        html += "</ol>"
        html += "</div>"
        return html

    # Handle dictionaries
    elif isinstance(field_value, dict):
        json_str = json.dumps(field_value, indent=2)
        return f"""
<div style='margin-bottom: 1rem;'>
    <h4 style='color: #667eea;'>📋 {formatted_name}</h4>
    <details>
        <summary style='cursor: pointer; color: #667eea; padding: 0.5rem; background: #f8f9fa; border-radius: 4px;'>
            Show Details
        </summary>
        <div style='background: #f8f9fa; padding: 1rem; margin-top: 0.5rem; border-radius: 4px; overflow-x: auto;'>
            <pre style='margin: 0; font-size: 0.85em;'>{json_str}</pre>
        </div>
    </details>
</div>
"""

    # Handle simple strings
    elif isinstance(field_value, str):
        return f"""
<div style='margin-bottom: 1rem;'>
    <h4 style='color: #667eea;'>📄 {formatted_name}</h4>
    <div style='background: white; padding: 1rem; border-left: 4px solid #667eea; border-radius: 4px;'>
        {field_value}
    </div>
</div>
"""

    # Handle numbers and booleans
    elif isinstance(field_value, (int, float, bool)):
        return f"<p><strong style='color: #667eea;'>{formatted_name}:</strong> {field_value}</p>"

    # Default fallback
    else:
        return f"<p><strong style='color: #667eea;'>{formatted_name}:</strong> {str(field_value)}</p>"


def create_record_detail(record: Dict, record_idx: int) -> str:
    """Create a dataset-agnostic detailed view for a single record.

    Special handling is applied for fields in SPECIAL_FIELDS (question, sut_answer, metrics).
    Other fields are rendered using `render_field_value`. Fields in EXCLUDE_FIELDS are ignored.
    """
    # Prefer transformed question for instruction-following datasets
    question = (
       record.get("question", f"Question {record_idx}")
    )

    html = f"""
<div style='background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem;'>
    <h3 style='margin-top: 0; color: #667eea;'>📝 Question {record_idx}</h3>
    <p style='font-size: 1.1em; line-height: 1.6;'>{question}</p>
</div>
"""

    # System Answer (if present) — render in a preformatted block to preserve formatting
    if "sut_answer" in record:
        html += f"""
<div style='margin-bottom: 1rem;'>
    <h4 style='color: #667eea;'>🤖 System Answer</h4>
    <div style='background: white; padding: 1rem; border-left: 4px solid #667eea; border-radius: 4px;'>
        <pre style='white-space: pre-wrap; font-family: inherit; margin: 0;'>{record["sut_answer"]}</pre>
    </div>
</div>
"""

    # Render remaining user-visible fields (excluding special and excluded fields)
    for field_name, field_value in record.items():
        if field_name not in SPECIAL_FIELDS and field_name not in EXCLUDE_FIELDS:
            html += render_field_value(field_name, field_value)

    # Metrics Details (if present)
    if "metrics" in record and isinstance(record["metrics"], dict):
        html += "<h4 style='color: #667eea;'>📊 Metric Details</h4>"
        for metric_name, metric_info in record["metrics"].items():
            if isinstance(metric_info, dict):
                score = metric_info.get("score", "N/A")
                summary = metric_info.get("explanation_summary", "")
                metadata = metric_info.get("explanation_metadata", {})

                # Display score without color coding
                if isinstance(score, (int, float)):
                    score_display = f"{score:.2f}"
                else:
                    score_display = str(score)

                html += f"""
<div style='background: white; padding: 1rem; margin-bottom: 1rem; border-radius: 8px; border-left: 4px solid {DEFAULT_CHART_COLOR};'>
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
        <strong style='font-size: 1.1em;'>{metric_name.replace("_", " ").title()}</strong>
        <span style='background: {DEFAULT_CHART_COLOR}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: bold;'>
            {score_display}
        </span>
    </div>
    <p style='margin: 0.5rem 0; color: #666; font-size: 0.9em;'>{summary}</p>
"""

                # Add expandable metadata section if metadata exists
                if metadata:
                    metadata_json = json.dumps(metadata, indent=2)
                    html += f"""
    <details style='margin-top: 0.75rem;'>
        <summary style='cursor: pointer; color: #667eea; font-weight: 600; padding: 0.5rem; background: #f8f9fa; border-radius: 4px;'>
            🔍 Show Explanation Metadata
        </summary>
        <div style='margin-top: 0.5rem; background: #f8f9fa; padding: 1rem; border-radius: 4px; overflow-x: auto;'>
            <pre style='margin: 0; font-size: 0.85em; white-space: pre-wrap;'>{metadata_json}</pre>
        </div>
    </details>
"""

                html += "</div>"

    return html


def load_metric_data(
    metric_file: str = "detailed_results_by_metric.json",
    output_file: str = "output.json",
):
    with open(metric_file) as f:
        metric_data = json.load(f)

    output_data = None
    try:
        with open(output_file) as f:
            output_data = json.load(f)
    except FileNotFoundError:
        pass

    return metric_data, output_data


def create_score_distribution_chart(scores: List[float], metric_name: str):
    """Create a histogram of sample scores for a metric."""
    if not scores:
        return None

    # Clip scores to [0,1] and create 10 fixed bins from 0.0 to 1.0 (size 0.1)
    clipped = [min(max(float(s), 0.0), 1.0) for s in scores]

    # Decide whether metric is binary or continuous
    is_binary = metric_name in BINARY_METRICS
    # If not explicitly defined, auto-detect binary when every value is exactly 0.0 or 1.0
    if not is_binary:
        if all(v in (0.0, 1.0) for v in clipped):
            is_binary = True

    fig = go.Figure()

    if is_binary:
        # Binary metric: two bars for 0 and 1
        zero_count = sum(1 for v in clipped if v < 0.5)
        one_count = sum(1 for v in clipped if v >= 0.5)
        labels = ["0", "1"]
        counts = [zero_count, one_count]

        fig.add_trace(
            go.Bar(
                x=labels,
                y=counts,
                marker_color=DEFAULT_CHART_COLOR,
                showlegend=False,
                hovertemplate="Score: %{x}<br>Count: %{y}<extra></extra>",
                text=[str(c) if c > 0 else "" for c in counts],
                textposition="outside",
            )
        )

        # Annotate mean value (numeric) above the bars
        mean_score = sum(scores) / len(scores)
        max_count = max(counts) if counts else 0
        fig.add_annotation(
            dict(
                x="1",
                y=max_count + max(1, int(0.05 * max_count)),
                xref="x",
                yref="y",
                text=f"Mean {mean_score:.2f}",
                showarrow=False,
                font=dict(size=11),
            )
        )

        xaxis = dict(tickmode="array", tickvals=labels)
        xaxis_title = "Score (0 or 1)"

    else:
        # Continuous metric: Compute fixed 10 bins [0.0-0.1), ..., [0.9-1.0]
        bin_count = 10
        counts = [0] * bin_count
        for v in clipped:
            # map 1.0 into last bin index
            idx = min(int(v * bin_count), bin_count - 1)
            counts[idx] += 1

        # Labels for bins
        labels = [f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(bin_count)]

        fig.add_trace(
            go.Bar(
                x=labels,
                y=counts,
                marker_color=DEFAULT_CHART_COLOR,
                showlegend=False,
                hovertemplate="Score range: %{x}<br>Count: %{y}<extra></extra>",
                text=[str(c) if c > 0 else "" for c in counts],
                textposition="outside",
            )
        )

        # Add mean marker at corresponding bin
        mean_score = sum(scores) / len(scores)
        mean_idx = min(int(mean_score * bin_count), bin_count - 1)
        mean_label = labels[mean_idx]
        fig.add_shape(
            dict(
                type="line",
                x0=mean_label,
                x1=mean_label,
                y0=0,
                y1=1,
                yref="paper",
                xref="x",
                line=dict(color="#495057", dash="dash"),
            )
        )
        fig.add_annotation(
            dict(
                x=mean_label,
                y=1.02,
                xref="x",
                yref="paper",
                text=f"Mean {mean_score:.2f}",
                showarrow=False,
                font=dict(size=11),
            )
        )

        xaxis = dict(tickmode="array", tickvals=labels)
        xaxis_title = "Score bins"

    fig.update_layout(
        title=f"{metric_name.replace('_', ' ').title()} — Score Distribution",
        xaxis_title=xaxis_title,
        yaxis_title="Count",
        xaxis=xaxis,
        template="plotly_white",
        height=450,
        margin=dict(t=70, b=40),
    )

    return fig


def create_metric_section(metric_name: str, records: List[Dict]) -> ar.Blocks:
    components = []

    # Header
    components.append(ar.Text(f"# 🧭 Metric: {metric_name.replace('_', ' ').title()}"))

    # Metric description
    desc = LAYER1_METRIC_DESCRIPTIONS.get(metric_name, "No description available.")
    components.append(
        ar.HTML(
            f"<div style='background: white; padding: 1rem; border: 1px solid #e9ecef; border-radius: 8px; margin: 1rem 0;'><strong>{metric_name.replace('_', ' ').title()}:</strong> {desc}</div>"
        )
    )

    # Extract numeric scores for distribution
    scores = []
    table_rows = []
    for idx, rec in enumerate(records, 1):
        score = None
        if "metrics" in rec and isinstance(rec["metrics"], dict):
            m_info = rec["metrics"].get(metric_name)
            if isinstance(m_info, dict):
                score = m_info.get("score")
        # allow numeric convertible
        try:
            if isinstance(score, (int, float)):
                scores.append(float(score))
        except Exception:
            pass

        # Row for table
        row = {
            "Record #": idx,
            "Question": rec.get("question")
            or rec.get("transformed_question", "")
            or rec.get("original_question", ""),
            "Score": score if score is not None else "",
        }
        table_rows.append(row)

    # Chart
    components.append(ar.Text("## 📈 Score Distribution"))
    chart = create_score_distribution_chart(scores, metric_name)
    if chart:
        components.append(ar.Plot(chart))
    else:
        components.append(ar.Text("*No numeric scores available for this metric.*"))

    # Table
    df = pd.DataFrame(table_rows)
    components.append(ar.Text("## 📊 Records"))
    components.append(ar.DataTable(df))

    # Detailed records (selectable)
    record_blocks = []
    for idx, rec in enumerate(records, 1):
        # Use dataset-agnostic record renderer
        detail_html = create_record_detail(rec, idx)
        record_blocks.append(ar.Group(ar.HTML(detail_html), label=f"Record {idx}"))

    if record_blocks:
        components.append(ar.Text("## 📄 Detailed Records"))
        # Use Select only if there are 2+ records; otherwise add directly
        if len(record_blocks) >= 2:
            components.append(ar.Select(blocks=record_blocks))
        else:
            components.extend(record_blocks)

    return ar.Group(*components, columns=1)


def create_report_by_metric(
    metric_data: Dict[str, List[Dict]], output_data: Dict[str, Any] = None
) -> ar.Blocks:
    # Overall header
    header = ar.Group(
        ar.Text(
            "# 📊 Accuracy Quality Indicator — Per-Metric Report"
        ),
        columns=1,
    )

    # Understanding Your Evaluation (client-facing guide)
    # Build sample-level metrics list dynamically from LAYER1_METRIC_DESCRIPTIONS with numbering
    sample_level_metrics_html = "<ol style='color: #6c757d; margin-left: 1.5rem; line-height: 2;'>"
    for idx, (metric_key, metric_desc) in enumerate(LAYER1_METRIC_DESCRIPTIONS.items(), 1):
        metric_display_name = metric_key.replace("_", " ").title()
        sample_level_metrics_html += f"<li><strong>{metric_display_name}:</strong> {metric_desc}</li>"
    sample_level_metrics_html += "</ol>"
    
    # Build system-level metrics list dynamically from LAYER2_METRIC_DESCRIPTIONS with numbering
    system_level_metrics_html = "<ol style='color: #6c757d; margin-left: 1.5rem; line-height: 2;'>"
    for idx, (metric_key, metric_desc) in enumerate(LAYER2_METRIC_DESCRIPTIONS.items(), 1):
        metric_display_name = metric_key.replace("_", " ").title()
        system_level_metrics_html += f"<li><strong>{metric_display_name}:</strong> {metric_desc}</li>"
    system_level_metrics_html += "</ol>"
    
    quick_start_intro = ar.HTML(f"""
<div style='background: #f8f9fa; border: 1px solid #e9ecef; padding: 2rem; border-radius: 12px; margin: 2rem 0;'>
    <h2 style='margin-top: 0; color: #495057;'>🎯 Understanding Your Evaluation</h2>
    
    <h3 style='color: #495057;'>Metrics Overview</h3>
    <p style='margin-bottom: 1.5rem; color: #6c757d;'>Your accuracy evaluation consists of <strong>{len(LAYER1_METRIC_DESCRIPTIONS)} sample-level metrics</strong> that measure individual sample quality, and <strong>{len(LAYER2_METRIC_DESCRIPTIONS)} system-level (aggregated) metrics</strong> that assess overall accuracy performance.</p>
    
    <h4 style='color: #495057; margin-top: 1.5rem; margin-bottom: 0.75rem;'>📊 Sample-Level Metrics (Measured Per Sample)</h4>
    {sample_level_metrics_html}
    
    <h4 style='color: #495057; margin-top: 2rem; margin-bottom: 0.75rem;'>📈 System-Level Metrics (Aggregated)</h4>
    <p style='color: #6c757d; font-size: 0.95em; margin-bottom: 0.75rem;'>Computed by aggregating sample-level metrics across all samples:</p>
    {system_level_metrics_html}
    
    <div style='background: #e7f3ff; padding: 1rem; border-radius: 6px; margin-top: 1.5rem; border-left: 4px solid #0066cc;'>
        <p style='margin: 0; color: #495057; font-size: 0.9em;'>
            <strong>💡 Note:</strong> All the system-level metrics are available for use in the scorecard to derive a system score and system grade (A to E).
        </p>
    </div>
""")

    quick_start_navigation = ar.HTML("""
    <h3 style='color: #495057; margin-top: 2rem;'>How to Navigate This Report</h3>
    <ol style='line-height: 1.8; color: #6c757d;'>
        <li><strong style='color: #495057;'>Check aggregated metric scores</strong> in the chart below</li>
        <li><strong style='color: #495057;'>Select individual metrics</strong> to view detailed score distributions</li>
        <li><strong style='color: #495057;'>Drill down into records</strong> to see detailed analysis for each question</li>
        <li><strong style='color: #495057;'>Look for patterns</strong> in failures to identify improvement areas</li>
    </ol>
    
    <p style='margin-bottom: 0; font-size: 0.9em; color: #6c757d;'>
        💡 <strong style='color: #495057;'>Tip:</strong> Use the data tables to sort and filter records by metric scores. Click on column headers to reorder.
    </p>
</div>
""")
    # Metric derivation guide (now merged into quick_start_intro above)

    # Create pages: one page per metric in LAYER1_METRIC_DESCRIPTIONS order
    pages = []
    metric_averages = {}  # Track average scores for overview chart
    
    # Iterate through LAYER1_METRIC_DESCRIPTIONS to maintain consistent ordering
    for metric_name in LAYER1_METRIC_DESCRIPTIONS.keys():
        if metric_name in metric_data:
            records = metric_data[metric_name]
            if isinstance(records, list):
                section = create_metric_section(metric_name, records)
                pages.append(ar.Group(section, label=metric_name.replace("_", " ").title()))
                
                # Calculate average score for this metric
                scores = []
                for rec in records:
                    if "metrics" in rec and isinstance(rec["metrics"], dict):
                        m_info = rec["metrics"].get(metric_name)
                        if isinstance(m_info, dict):
                            score = m_info.get("score")
                            if isinstance(score, (int, float)):
                                scores.append(float(score))
                
                if scores:
                    metric_averages[metric_name] = sum(scores) / len(scores)
    
    # Create overview bar chart of system-level aggregated metrics from output_data
    metrics_overview_chart = None
    if output_data:
        # Define system-level metrics in display order (these are aggregated metrics)
        system_metrics_order = [
            ("answer_correctness", "Answer Correctness"),
            ("faithfulness", "Faithfulness"),
            ("helpfulness", "Helpfulness"),
            ("retrieval_correctness", "Retrieval Correctness"),
            ("hit@k", "Hit@K"),
            ("task_success", "Task Success"),
            ("retrieval_gate_pass_rate", "Retrieval Gate Pass Rate"),
            ("faithfulness_gate_pass_rate", "Faithfulness Gate Pass Rate"),
            ("hard_gate_pass_rate", "Hard Gate Pass Rate"),
            ("conditional_task_success", "Conditional Task Success"),
            ("instruction_following", "Instruction Following")
        ]
        
        # Extract values from output_data in defined order, sorted by value for display
        aggregated_metrics = []
        for key, display_name in system_metrics_order:
            if key in output_data:
                value = float(output_data[key])
                aggregated_metrics.append((display_name, value))
        
        # Sort by value for better visualization
        sorted_metrics = sorted(aggregated_metrics, key=lambda x: x[1])
        metric_labels = [m for m, _ in sorted_metrics]
        metric_values = [v for _, v in sorted_metrics]
        
        if metric_labels and metric_values:
            # Use neutral gray color for all bars
            fig = go.Figure(go.Bar(
                x=metric_values,
                y=metric_labels,
                orientation='h',
                marker_color=DEFAULT_CHART_COLOR,  # Neutral gray
                text=[f"{v:.2f}" for v in metric_values],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="📊 Aggregate System-Level Metrics Scores",
                xaxis_title="Aggregated Score",
                yaxis_title="Metric",
                template="plotly_white",
                height=max(400, len(metric_labels) * 40),
                margin=dict(l=200, r=40, t=60, b=40),
                xaxis=dict(range=[0, 1.0])
            )
            
            metrics_overview_chart = ar.Plot(fig)

    if pages:
        blocks = [header, quick_start_intro, quick_start_navigation]
        blocks.extend([
            ar.Text("---"),
            ar.Text("## 📂 System-Level Metrics"),
        ])
        
        # Add overview chart if available
        if metrics_overview_chart:
            blocks.extend([
                ar.Text("*These aggregated system-level metrics are computed by combining sample-level measurements. Lower scores indicate areas for improvement.*"),
                metrics_overview_chart,
                ar.Text("---"),
            ])
        
        # Use Select only if there are 2+ pages; otherwise add metrics directly
        if len(pages) >= 2:
            blocks.extend([
                ar.Text("## 🔍 Sample-Level Metrics"),
                ar.Text("*Select a metric below to view detailed score distribution and individual samples.*"),
                ar.Select(blocks=pages),
                ar.Text("---"),
                ar.Text("*Generated with Arakawa (metric view)*"),
            ])
        else:
            blocks.extend([
                ar.Text("*Detailed score distribution and individual samples below.*"),
                ar.Text("---"),
            ])
            blocks.extend(pages)
            blocks.append(ar.Text("*Generated with Arakawa (metric view)*"))
        
        report = ar.Blocks(*blocks)
    else:
        blocks = [header, quick_start_intro, quick_start_navigation]
        blocks.append(ar.Text("⚠️ No metric groups found."))
        report = ar.Blocks(*blocks)

    return report