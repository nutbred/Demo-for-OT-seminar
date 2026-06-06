from __future__ import annotations

import math

import numpy as np
import ot
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PRESETS = {
    "Bài seminar 3 x 4": {
        "cost": np.array(
            [
                [1, 8, 9, 6],
                [7, 2, 8, 3],
                [7, 8, 5, 4],
            ],
            dtype=float,
        ),
        "supply": np.array([20, 30, 25], dtype=float),
        "demand": np.array([20, 25, 20, 10], dtype=float),
    },
    "Mẫu lớn 5 x 6": {
        "cost": np.array(
            [
                [4, 8, 8, 6, 9, 7],
                [6, 4, 7, 5, 8, 6],
                [5, 7, 3, 8, 6, 4],
                [9, 6, 5, 4, 7, 8],
                [7, 5, 6, 3, 4, 5],
            ],
            dtype=float,
        ),
        "supply": np.array([25, 35, 20, 30, 40], dtype=float),
        "demand": np.array([20, 30, 25, 35, 15, 25], dtype=float),
    },
}
DEFAULT_PRESET = "Bài seminar 3 x 4"


def make_labels(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{index}" for index in range(1, count + 1)]


def cost_dataframe(values: np.ndarray, source_labels: list[str], dest_labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(values, index=source_labels, columns=dest_labels)


def vector_dataframe(label: str, labels: list[str], values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({label: values}, index=labels)


def numeric_frame(frame: pd.DataFrame, name: str) -> np.ndarray:
    numeric = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} chỉ được chứa số hữu hạn.")
    if np.any(numeric < 0):
        raise ValueError(f"{name} không được âm.")
    return numeric


def extract_inputs(
    cost_df: pd.DataFrame,
    supply_df: pd.DataFrame,
    demand_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    cost = numeric_frame(cost_df, "Ma trận chi phí")
    supply_matrix = numeric_frame(supply_df, "Vector cung")
    demand_matrix = numeric_frame(demand_df, "Vector cầu")

    if supply_matrix.shape[1] != 1:
        raise ValueError("Vector cung phải có đúng 1 cột.")
    if demand_matrix.shape[1] != 1:
        raise ValueError("Vector cầu phải có đúng 1 cột.")

    supply = supply_matrix.reshape(-1)
    demand = demand_matrix.reshape(-1)
    source_labels = [str(label) for label in cost_df.index]
    dest_labels = [str(label) for label in cost_df.columns]

    if cost.shape != (len(supply), len(demand)):
        raise ValueError(
            "Kích thước chưa khớp: ma trận chi phí phải có số hàng bằng số nguồn "
            "và số cột bằng số điểm nhận."
        )

    total_supply = float(np.sum(supply))
    total_demand = float(np.sum(demand))
    if total_supply <= 0 or total_demand <= 0:
        raise ValueError("Tổng cung và tổng cầu phải lớn hơn 0.")
    if not np.isclose(total_supply, total_demand, rtol=1e-9, atol=1e-9):
        raise ValueError(f"Bài toán chưa cân bằng: tổng cung = {total_supply:g}, tổng cầu = {total_demand:g}.")

    return cost, supply, demand, source_labels, dest_labels


def solve_exact_ot(supply: np.ndarray, demand: np.ndarray, cost: np.ndarray) -> np.ndarray:
    return np.asarray(ot.emd(supply.astype(float), demand.astype(float), cost.astype(float)))


def store_solution(
    cost: np.ndarray,
    supply: np.ndarray,
    demand: np.ndarray,
    source_labels: list[str],
    dest_labels: list[str],
) -> None:
    plan = solve_exact_ot(supply, demand, cost)
    st.session_state.solution = {
        "cost": cost,
        "supply": supply,
        "demand": demand,
        "source_labels": source_labels,
        "dest_labels": dest_labels,
        "plan": plan,
        "total_cost": float(np.sum(plan * cost)),
    }
    st.session_state.solution_error = None


def load_preset(preset_name: str) -> None:
    preset = PRESETS[preset_name]
    cost = preset["cost"].copy()
    supply = preset["supply"].copy()
    demand = preset["demand"].copy()
    source_labels = make_labels("S", cost.shape[0])
    dest_labels = make_labels("D", cost.shape[1])

    st.session_state.active_preset = preset_name
    st.session_state.cost_df = cost_dataframe(cost, source_labels, dest_labels)
    st.session_state.supply_df = vector_dataframe("Cung", source_labels, supply)
    st.session_state.demand_df = vector_dataframe("Cầu", dest_labels, demand)
    st.session_state.input_version = st.session_state.get("input_version", 0) + 1
    store_solution(cost, supply, demand, source_labels, dest_labels)


def ensure_session_defaults() -> None:
    required_keys = ("active_preset", "cost_df", "supply_df", "demand_df", "input_version", "solution")
    if any(key not in st.session_state for key in required_keys):
        load_preset(DEFAULT_PRESET)


def commit_editor_inputs(cost_df: pd.DataFrame, supply_df: pd.DataFrame, demand_df: pd.DataFrame) -> None:
    cost, supply, demand, source_labels, dest_labels = extract_inputs(cost_df, supply_df, demand_df)
    st.session_state.cost_df = cost_dataframe(cost, source_labels, dest_labels)
    st.session_state.supply_df = vector_dataframe("Cung", source_labels, supply)
    st.session_state.demand_df = vector_dataframe("Cầu", dest_labels, demand)
    store_solution(cost, supply, demand, source_labels, dest_labels)


def format_number(value: float, digits: int) -> str:
    if abs(value) < 10 ** (-(digits + 2)):
        value = 0.0
    if digits == 0:
        return f"{value:.0f}"
    return f"{value:.{digits}f}"


def display_dataframe(values: np.ndarray, rows: list[str], cols: list[str], digits: int) -> pd.DataFrame:
    return pd.DataFrame(
        [[format_number(float(value), digits) for value in row] for row in values],
        index=rows,
        columns=cols,
    )


def frames_have_same_numbers(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left.index) != list(right.index) or list(left.columns) != list(right.columns):
        return False
    try:
        left_values = left.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        right_values = right.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    except ValueError:
        return False
    return np.allclose(left_values, right_values, rtol=1e-9, atol=1e-9, equal_nan=True)


def has_uncommitted_changes(cost_df: pd.DataFrame, supply_df: pd.DataFrame, demand_df: pd.DataFrame) -> bool:
    return not (
        frames_have_same_numbers(cost_df, st.session_state.cost_df)
        and frames_have_same_numbers(supply_df, st.session_state.supply_df)
        and frames_have_same_numbers(demand_df, st.session_state.demand_df)
    )


def make_heatmap(plan: np.ndarray, source_labels: list[str], dest_labels: list[str], digits: int) -> go.Figure:
    annotations = [[format_number(float(value), digits) for value in row] for row in plan]
    figure = go.Figure(
        data=go.Heatmap(
            z=plan,
            x=dest_labels,
            y=source_labels,
            text=annotations,
            texttemplate="%{text}",
            colorscale="YlGnBu",
            hovertemplate="Từ %{y} đến %{x}<br>Lượng: %{z}<extra></extra>",
            colorbar={"title": "Lượng"},
        )
    )
    figure.update_layout(
        height=max(340, 55 * len(source_labels)),
        margin={"l": 45, "r": 30, "t": 30, "b": 40},
        xaxis_title="Điểm nhận",
        yaxis_title="Nguồn",
    )
    return figure


def evenly_spaced(count: int) -> list[float]:
    if count == 1:
        return [0.5]
    return list(np.linspace(0.9, 0.1, count))


def make_transport_graph(
    plan: np.ndarray,
    source_labels: list[str],
    dest_labels: list[str],
    threshold: float,
    digits: int,
) -> go.Figure:
    max_flow = float(np.max(plan)) if plan.size else 0.0
    source_y = evenly_spaced(len(source_labels))
    dest_y = evenly_spaced(len(dest_labels))

    node_x = [0.08] * len(source_labels) + [0.92] * len(dest_labels)
    node_y = source_y + dest_y
    node_text = source_labels + dest_labels

    figure = go.Figure()

    for i, source in enumerate(source_labels):
        for j, dest in enumerate(dest_labels):
            flow = float(plan[i, j])
            if flow <= threshold:
                continue
            width = 1.5 if max_flow <= 0 else 1.5 + 8.5 * math.sqrt(flow / max_flow)
            figure.add_trace(
                go.Scatter(
                    x=[0.12, 0.88],
                    y=[source_y[i], dest_y[j]],
                    mode="lines",
                    line={"width": width, "color": "rgba(31, 119, 180, 0.55)"},
                    hovertemplate=(
                        f"{source} -> {dest}<br>"
                        f"Lượng: {format_number(flow, digits)}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

    figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker={
                "size": 34,
                "color": ["#2f7d32"] * len(source_labels) + ["#b23a48"] * len(dest_labels),
                "line": {"width": 1.5, "color": "white"},
            },
            text=node_text,
            textposition="middle center",
            textfont={"color": "white", "size": 13},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    figure.update_layout(
        height=max(360, 62 * max(len(source_labels), len(dest_labels))),
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        xaxis={"visible": False, "range": [0, 1]},
        yaxis={"visible": False, "range": [0, 1]},
        plot_bgcolor="white",
    )
    return figure


st.set_page_config(page_title="Demo Optimal Transport", page_icon="OT", layout="wide")
ensure_session_defaults()
input_version = st.session_state.input_version

st.title("Demo Optimal Transport")
st.caption("Demo bài toán vận tải cân bằng bằng exact OT của POT.")

with st.sidebar:
    st.header("Điều khiển")
    selected_preset = st.selectbox(
        "Bài mẫu",
        list(PRESETS.keys()),
        index=list(PRESETS.keys()).index(st.session_state.active_preset),
    )
    if st.button("Tải / reset bài mẫu", width="stretch"):
        load_preset(selected_preset)
        st.rerun()

    digits_label = st.selectbox(
        "Hiển thị số",
        ["Số nguyên", "2 chữ số thập phân", "4 chữ số thập phân"],
        index=0,
    )
    digits = {"Số nguyên": 0, "2 chữ số thập phân": 2, "4 chữ số thập phân": 4}[digits_label]

    edge_threshold = st.number_input(
        "Ngưỡng hiện cạnh",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="Chỉ vẽ các tuyến có lượng vận chuyển lớn hơn ngưỡng này.",
    )

st.subheader("Dữ liệu đầu vào")
left, middle, right = st.columns([2.5, 1, 1])

with left:
    st.markdown("**Ma trận chi phí C**")
    cost_df = st.data_editor(
        st.session_state.cost_df,
        key=f"cost_editor_{input_version}",
        num_rows="fixed",
        width="stretch",
        column_config={
            col: st.column_config.NumberColumn(col, min_value=0.0, step=1.0, format="%.4f")
            for col in st.session_state.cost_df.columns
        },
    )

with middle:
    st.markdown("**Cung s**")
    supply_df = st.data_editor(
        st.session_state.supply_df,
        key=f"supply_editor_{input_version}",
        num_rows="fixed",
        width="stretch",
        column_config={"Cung": st.column_config.NumberColumn("Cung", min_value=0.0, step=1.0, format="%.4f")},
    )

with right:
    st.markdown("**Cầu d**")
    demand_df = st.data_editor(
        st.session_state.demand_df,
        key=f"demand_editor_{input_version}",
        num_rows="fixed",
        width="stretch",
        column_config={"Cầu": st.column_config.NumberColumn("Cầu", min_value=0.0, step=1.0, format="%.4f")},
    )

action_left, action_right = st.columns([1, 3])
with action_left:
    calculate_clicked = st.button("Tính lại", type="primary", width="stretch")

if calculate_clicked:
    try:
        commit_editor_inputs(cost_df, supply_df, demand_df)
        st.success("Đã tính lại output từ input hiện tại.")
    except ValueError as exc:
        st.session_state.solution = None
        st.session_state.solution_error = str(exc)
    except Exception as exc:  # POT can raise solver-specific errors for pathological input.
        st.session_state.solution = None
        st.session_state.solution_error = f"Không giải được bài toán với dữ liệu hiện tại: {exc}"
elif has_uncommitted_changes(cost_df, supply_df, demand_df):
    with action_right:
        st.warning("Input đã được chỉnh. Bấm **Tính lại** để cập nhật output.")

if st.session_state.solution_error:
    st.error(st.session_state.solution_error)
    st.info("Hãy chỉnh lại dữ liệu sao cho mọi giá trị không âm và tổng cung bằng tổng cầu.")
    st.stop()

solution = st.session_state.solution
if solution is None:
    st.info("Chưa có output hợp lệ. Hãy chỉnh input rồi bấm **Tính lại**.")
    st.stop()

cost = solution["cost"]
supply = solution["supply"]
demand = solution["demand"]
source_labels = solution["source_labels"]
dest_labels = solution["dest_labels"]
plan = solution["plan"]
total_cost = solution["total_cost"]

st.subheader("Kết quả exact OT")
metric_cols = st.columns(3)
metric_cols[0].metric("Tổng cung", format_number(float(np.sum(supply)), digits))
metric_cols[1].metric("Tổng cầu", format_number(float(np.sum(demand)), digits))
metric_cols[2].metric("Chi phí tối ưu", format_number(total_cost, digits))

result_left, result_right = st.columns([1.3, 1])
with result_left:
    st.markdown("**Transport plan P**")
    st.dataframe(
        display_dataframe(plan, source_labels, dest_labels, digits),
        width="stretch",
    )

with result_right:
    nonzero_routes = int(np.sum(plan > edge_threshold))
    st.markdown("**Tóm tắt**")
    st.write(f"Kích thước bài toán: **{len(source_labels)} x {len(dest_labels)}**")
    st.write(f"Số tuyến đang hiển thị trên graph: **{nonzero_routes}**")
    st.write("Objective: **sum(P * C)**")

viz_left, viz_right = st.columns(2)
with viz_left:
    st.markdown("**Heatmap lượng vận chuyển**")
    st.plotly_chart(make_heatmap(plan, source_labels, dest_labels, digits), width="stretch")

with viz_right:
    st.markdown("**Graph nguồn - điểm nhận**")
    st.plotly_chart(make_transport_graph(plan, source_labels, dest_labels, edge_threshold, digits), width="stretch")
