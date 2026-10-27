import warnings
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

DATASET_ID = 891
TARGET = "Diabetes_binary"
UCI_URL = "https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators"
UCI_DOI = "https://doi.org/10.24432/C53919"

st.set_page_config(
    page_title="당뇨병 데이터 AI 성능 탐구",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1280px; padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #f4f7fb;
        border: 1px solid #e2e8f0;
        padding: 12px 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧬 당뇨병 건강지표 데이터 기반 머신러닝 분석")
st.caption("탐구 질문: 전체 정확도는 성별·연령대별 신뢰성을 보장하는가?")
st.warning(
    "교육·탐구용 프로그램입니다. 공개 설문 데이터의 분류 성능을 실험하는 도구이며, "
    "개인의 당뇨병 진단이나 미래 발병 예측에 사용하면 안 됩니다."
)


@st.cache_data(show_spinner="UCI 공개 데이터셋을 불러오는 중입니다...")
def load_data():
    """UCI 공식 저장소에서 데이터셋을 불러온다."""
    from ucimlrepo import fetch_ucirepo

    repository = fetch_ucirepo(id=DATASET_ID)
    features = repository.data.features.copy()
    targets = repository.data.targets.copy() if repository.data.targets is not None else None
    features.columns = [str(c).strip() for c in features.columns]

    target_values = None
    feature_target = next(
        (c for c in features.columns if c.lower() == TARGET.lower()), None
    )

    if feature_target:
        target_values = features.pop(feature_target).reset_index(drop=True)
    elif targets is not None:
        targets.columns = [str(c).strip() for c in targets.columns]
        target_col = next(
            (c for c in targets.columns if c.lower() == TARGET.lower()), None
        )
        if target_col is None and targets.shape[1] == 1:
            target_col = targets.columns[0]
        if target_col is not None:
            target_values = targets[target_col].reset_index(drop=True)

    if target_values is None:
        raise ValueError("당뇨병 결과 변수 Diabetes_binary를 찾을 수 없습니다.")
    if len(features) != len(target_values):
        raise ValueError("입력 데이터와 결과 데이터의 행 수가 일치하지 않습니다.")

    df = features.reset_index(drop=True).copy()
    df[TARGET] = pd.to_numeric(target_values, errors="coerce").to_numpy()
    df = df.dropna(subset=[TARGET]).copy()
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    for column in df.columns:
        if column != TARGET:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Sex" not in df.columns or "Age" not in df.columns:
        raise ValueError("데이터셋에 성별(Sex) 또는 연령(Age) 변수가 없습니다.")
    if df[TARGET].nunique() < 2:
        raise ValueError("결과 변수에 0과 1 두 분류가 모두 있어야 합니다.")
    return df


def age_band(value):
    """UCI의 Age 코드(1~13)를 분석용 연령대로 묶는다."""
    try:
        code = int(value)
    except (TypeError, ValueError):
        return "알 수 없음"
    if 1 <= code <= 4:
        return "18–39세"
    if 5 <= code <= 7:
        return "40–54세"
    if 8 <= code <= 10:
        return "55–69세"
    if 11 <= code <= 13:
        return "70세 이상"
    return "알 수 없음"


def safe_percent(numerator, denominator):
    return float(100 * numerator / denominator) if denominator else np.nan


def calculate_metrics(y_true, y_pred, scores=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    result = {
        "평가 표본 수": int(len(y_true)),
        "실제 양성 수": int((np.asarray(y_true) == 1).sum()),
        "정확도(%)": 100 * accuracy_score(y_true, y_pred),
        "민감도(%)": safe_percent(tp, tp + fn),
        "특이도(%)": safe_percent(tn, tn + fp),
        "위음성률(%)": safe_percent(fn, fn + tp),
        "위양성률(%)": safe_percent(fp, fp + tn),
        "ROC-AUC": roc_auc_score(y_true, scores)
        if scores is not None and len(np.unique(y_true)) == 2
        else np.nan,
    }
    return result, cm


@st.cache_data(show_spinner=False)
def train_and_evaluate(sample_size, selected_features, model_choice,
                       test_fraction, threshold, rf_trees, rf_depth, random_seed):
    df = load_data()

    if sample_size < len(df):
        working_df, _ = train_test_split(
            df, train_size=int(sample_size), stratify=df[TARGET],
            random_state=int(random_seed)
        )
    else:
        working_df = df.copy()

    train_df, test_df = train_test_split(
        working_df, test_size=float(test_fraction), stratify=working_df[TARGET],
        random_state=int(random_seed)
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    feature_cols = list(selected_features)
    if not feature_cols:
        raise ValueError("입력 변수를 한 개 이상 선택해 주세요.")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET].to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET].to_numpy()

    model_names = (
        ["로지스틱 회귀", "랜덤 포레스트"]
        if model_choice == "두 모델 비교" else [model_choice]
    )

    test_groups = test_df.copy()
    test_groups["성별"] = test_groups["Sex"].map({0: "여성", 1: "남성"}).fillna("알 수 없음")
    test_groups["연령대"] = test_groups["Age"].apply(age_band)

    overall_rows, group_rows = [], []
    confusion_data, trained_models, profile_defaults = {}, {}, {}

    for column in feature_cols:
        values = train_df[column].dropna()
        if values.empty:
            profile_defaults[column] = 0.0
        elif column in ("Sex", "Age") or values.nunique() <= 15:
            profile_defaults[column] = float(values.mode().iloc[0])
        else:
            profile_defaults[column] = float(values.median())

    for name in model_names:
        if name == "로지스틱 회귀":
            estimator = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=600, class_weight="balanced",
                    solver="liblinear", random_state=int(random_seed)
                )),
            ])
        else:
            estimator = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    n_estimators=int(rf_trees), max_depth=int(rf_depth),
                    min_samples_leaf=10, class_weight="balanced_subsample",
                    n_jobs=-1, random_state=int(random_seed)
                )),
            ])

        estimator.fit(X_train, y_train)
        positive_index = list(estimator.classes_).index(1)
        scores = estimator.predict_proba(X_test)[:, positive_index]
        predictions = (scores >= float(threshold)).astype(int)

        overall, cm = calculate_metrics(y_test, predictions, scores)
        overall_rows.append({"모델": name, **overall})
        confusion_data[name] = cm.tolist()
        trained_models[name] = estimator

        scored = test_groups.copy()
        scored["실제값"] = y_test
        scored["예측값"] = predictions
        scored["양성 점수"] = scores

        for column, ordered_groups in [
            ("성별", ["여성", "남성", "알 수 없음"]),
            ("연령대", ["18–39세", "40–54세", "55–69세", "70세 이상", "알 수 없음"]),
        ]:
            for group_name in ordered_groups:
                group = scored[scored[column] == group_name]
                if group.empty:
                    continue
                group_metrics, _ = calculate_metrics(
                    group["실제값"].to_numpy(),
                    group["예측값"].to_numpy(),
                    group["양성 점수"].to_numpy(),
                )
                group_rows.append({
                    "모델": name, "집단 기준": column, "집단": group_name,
                    **group_metrics
                })

    info = {
        "전체 데이터 수": len(df),
        "사용 표본 수": len(working_df),
        "학습 표본 수": len(train_df),
        "평가 표본 수": len(test_df),
        "입력 변수 수": len(feature_cols),
        "평가 양성 수": int((y_test == 1).sum()),
        "평가 음성 수": int((y_test == 0).sum()),
    }
    return (
        pd.DataFrame(overall_rows), pd.DataFrame(group_rows),
        confusion_data, trained_models, profile_defaults, info
    )


def percent_column_config():
    return {
        "정확도(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "민감도(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "특이도(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "위음성률(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "위양성률(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
    }


try:
    data = load_data()
except Exception as error:
    st.error("UCI 데이터셋을 불러오지 못했습니다. 인터넷 연결을 확인하고 다시 실행해 주세요.")
    st.markdown(f"[UCI 공식 데이터셋 페이지]({UCI_URL})")
    with st.expander("오류 세부 내용"):
        st.code(str(error))
    st.stop()

all_features = [c for c in data.columns if c != TARGET and c != "ID"]

a, b, c, d = st.columns(4)
a.metric("전체 관측값", f"{len(data):,}개")
b.metric("입력 변수", f"{len(all_features)}개")
c.metric("양성 범주 비율", f"{(data[TARGET] == 1).mean() * 100:.1f}%")
d.metric("결과 변수", "이진 분류")

with st.expander("데이터 설명"):
    st.markdown(f"""
    **출처:** [UCI Machine Learning Repository — CDC Diabetes Health Indicators]({UCI_URL})

    - `Diabetes_binary = 0`: 데이터셋의 비당뇨 범주
    - `Diabetes_binary = 1`: 데이터셋의 전당뇨/당뇨병 범주
    - `Sex`: 성별 코드(0=여성, 1=남성)
    - `Age`: 실제 나이가 아니라 연령대 코드(1~13)
    - `BMI`, `HighBP`, `HighChol`, `PhysActivity` 등 건강·생활습관 변수 포함

    이 자료는 미국의 2015년 건강행태 설문 자료를 기반으로 하며,
    미래 발병 여부나 개인의 임상 진단을 직접 나타내는 자료는 아닙니다.
    """)

left, right = st.columns([1, 1])
with left:
    st.subheader("분류 라벨 분포")
    label_counts = (
        data[TARGET].map({0: "0 · 비당뇨 범주", 1: "1 · 전당뇨/당뇨병 범주"})
        .value_counts().rename_axis("분류").to_frame("표본 수")
    )
    st.bar_chart(label_counts)
with right:
    st.subheader("데이터 미리보기")
    st.dataframe(data.head(10), use_container_width=True, hide_index=True)

st.divider()
st.header("1. 직접 조절할 분석 변수와 모델 설정")
st.write(
    "왼쪽 설정에서 학습 변수, 표본 수, 평가 비율, 양성 판정 임계값 등을 바꿀 수 있어. "
    "결과 차이를 해석하기 쉽게 한 번에 한 설정씩 바꿔 비교해 봐."
)

with st.sidebar:
    st.header("분석 설정")
    with st.form("analysis_settings"):
        sample_max = len(data)
        sample_min = min(5000, sample_max)
        sample_default = min(50000, sample_max)
        sample_size = st.slider(
            "분석 표본 수", min_value=sample_min, max_value=sample_max,
            value=sample_default,
            step=5000 if sample_max > 10000 else 1,
            help="선택한 표본을 층화 추출해 학습과 평가에 사용합니다."
        )
        selected_features = st.multiselect(
            "모델에 넣을 입력 변수", options=all_features, default=all_features,
            help="원하는 변수를 포함하거나 제외할 수 있습니다. 성별·연령대 평가 자체는 원본 변수를 기준으로 합니다."
        )
        model_choice = st.selectbox("모델", ["두 모델 비교", "로지스틱 회귀", "랜덤 포레스트"])
        test_fraction = st.slider("평가 데이터 비율", 0.10, 0.40, 0.25, 0.05)
        threshold = st.slider("양성 판정 임계값", 0.10, 0.90, 0.50, 0.05)
        rf_trees = st.slider("랜덤 포레스트 나무 개수", 50, 300, 100, 50)
        rf_depth = st.slider("랜덤 포레스트 최대 깊이", 2, 30, 14, 2)
        random_seed = st.number_input("무작위 시드", 0, 9999, 42, 1)
        submitted = st.form_submit_button("설정으로 분석 실행", type="primary", use_container_width=True)

if "analysis_config" not in st.session_state:
    st.session_state["analysis_config"] = {
        "sample_size": int(min(50000, len(data))),
        "selected_features": list(all_features),
        "model_choice": "두 모델 비교",
        "test_fraction": 0.25, "threshold": 0.50,
        "rf_trees": 100, "rf_depth": 14, "random_seed": 42,
    }
if submitted:
    if not selected_features:
        st.sidebar.error("입력 변수를 최소 1개 선택해 주세요.")
    else:
        st.session_state["analysis_config"] = {
            "sample_size": int(sample_size),
            "selected_features": list(selected_features),
            "model_choice": model_choice,
            "test_fraction": float(test_fraction),
            "threshold": float(threshold),
            "rf_trees": int(rf_trees),
            "rf_depth": int(rf_depth),
            "random_seed": int(random_seed),
        }

config = st.session_state["analysis_config"]
if not config["selected_features"]:
    st.error("왼쪽 메뉴에서 입력 변수를 한 개 이상 선택하고 분석을 실행해 주세요.")
    st.stop()

with st.spinner("모델을 학습하고 평가하고 있습니다..."):
    try:
        overall_df, group_df, confusion_data, trained_models, profile_defaults, info = train_and_evaluate(
            config["sample_size"], config["selected_features"], config["model_choice"],
            config["test_fraction"], config["threshold"], config["rf_trees"],
            config["rf_depth"], config["random_seed"]
        )
    except Exception as error:
        st.error("분석 중 오류가 발생했습니다. 표본 수나 선택 변수를 조절한 뒤 다시 실행해 주세요.")
        with st.expander("오류 세부 내용"):
            st.code(str(error))
        st.stop()

st.caption(
    f"현재 설정: 표본 {config['sample_size']:,}개 · 입력 변수 {len(config['selected_features'])}개 · "
    f"평가 비율 {config['test_fraction']:.0%} · 임계값 {config['threshold']:.2f}"
)

st.header("2. 전체 성능")
m1, m2, m3, m4 = st.columns(4)
m1.metric("학습 표본", f"{info['학습 표본 수']:,}개")
m2.metric("평가 표본", f"{info['평가 표본 수']:,}개")
m3.metric("평가 양성 표본", f"{info['평가 양성 수']:,}개")
m4.metric("학습 입력 변수", f"{info['입력 변수 수']}개")

st.dataframe(overall_df, use_container_width=True, hide_index=True, column_config=percent_column_config())
st.caption(
    "민감도 = 실제 양성 중 양성으로 맞힌 비율 · 위음성률 = 실제 양성 중 음성으로 놓친 비율 · "
    "위양성률 = 실제 음성 중 양성으로 잘못 분류한 비율."
)

st.subheader("혼동행렬")
cm_cols = st.columns(len(confusion_data))
for col, (model_name, matrix) in zip(cm_cols, confusion_data.items()):
    with col:
        st.markdown(f"**{model_name}**")
        cm_df = pd.DataFrame(
            matrix,
            index=["실제 0 · 비당뇨", "실제 1 · 전당뇨/당뇨병"],
            columns=["예측 0", "예측 1"],
        )
        st.dataframe(cm_df, use_container_width=True)

st.divider()
st.header("3. 성별·연령대별 오류율 비교")
group_choice = st.radio("비교 집단", ["성별", "연령대"], horizontal=True)
metric_choice = st.selectbox(
    "그래프로 비교할 지표",
    ["위음성률(%)", "민감도(%)", "정확도(%)", "특이도(%)", "위양성률(%)"],
    index=0,
)
plot_df = group_df[group_df["집단 기준"] == group_choice].copy()
if plot_df.empty:
    st.info("선택한 집단의 비교 결과가 없습니다.")
else:
    chart_data = plot_df.pivot(index="집단", columns="모델", values=metric_choice)
    st.bar_chart(chart_data, y_label=metric_choice, use_container_width=True)

st.subheader("집단별 세부 지표")
st.dataframe(
    group_df[group_df["집단 기준"] == group_choice],
    use_container_width=True, hide_index=True,
    column_config=percent_column_config()
)

if not group_df.empty and (group_df["평가 표본 수"] < 100).any():
    st.warning(
        "평가 표본이 100개 미만인 집단이 있습니다. 소규모 집단에서는 몇 건의 오분류만으로도 "
        "오류율이 크게 바뀔 수 있으므로 주의해서 해석하세요."
    )

st.divider()
st.header("4. 가상 프로필의 입력값 직접 변경")
st.write(
    "아래에서 데이터셋의 범위 안에서 변수값을 직접 바꾸고, 학습된 모델의 분류 결과가 어떻게 달라지는지 확인해 봐. "
    "가상 프로필 실험이지 실제 건강 평가가 아니야."
)
st.warning("이 결과는 실제 개인의 당뇨병 진단이나 발병 확률로 해석하면 안 됩니다.")

AGE_LABELS = {
    1: "18–24세", 2: "25–29세", 3: "30–34세", 4: "35–39세",
    5: "40–44세", 6: "45–49세", 7: "50–54세", 8: "55–59세",
    9: "60–64세", 10: "65–69세", 11: "70–74세", 12: "75–79세", 13: "80세 이상",
}
GEN_HEALTH_LABELS = {1: "매우 좋음", 2: "좋음", 3: "보통", 4: "나쁨", 5: "매우 나쁨"}
SEX_LABELS = {0: "여성", 1: "남성"}


def profile_input(column, default_value, available_values):
    vals = [float(v) for v in available_values if pd.notna(v)]
    if not vals:
        return st.number_input(column, value=float(default_value), key=f"profile_{column}")
    unique_values = sorted(set(vals))

    if column == "Sex" and set(unique_values).issubset({0.0, 1.0}):
        return st.selectbox(
            "Sex · 성별", [0, 1],
            index=0 if int(round(default_value)) == 0 else 1,
            format_func=lambda x: SEX_LABELS[x], key=f"profile_{column}"
        )
    if column == "Age":
        options = sorted(int(v) for v in unique_values)
        default = int(round(default_value))
        return st.selectbox(
            "Age · 연령 구간",
            options,
            index=options.index(default) if default in options else 0,
            format_func=lambda x: f"{x} · {AGE_LABELS.get(x, '연령 구간')}",
            key=f"profile_{column}"
        )
    if column == "GenHlth":
        options = sorted(int(v) for v in unique_values)
        default = int(round(default_value))
        return st.selectbox(
            "GenHlth · 주관적 건강 상태",
            options,
            index=options.index(default) if default in options else 0,
            format_func=lambda x: f"{x} · {GEN_HEALTH_LABELS.get(x, '코드')}",
            key=f"profile_{column}"
        )

    if set(unique_values).issubset({0.0, 1.0}):
        default = int(round(default_value))
        return st.selectbox(
            f"{column} · 이진 변수", [0, 1],
            index=default if default in [0, 1] else 0,
            format_func=lambda x: "예 (1)" if x == 1 else "아니오 (0)",
            key=f"profile_{column}"
        )

    low, high = float(min(vals)), float(max(vals))
    current = float(np.clip(default_value, low, high))
    if all(float(v).is_integer() for v in unique_values):
        low_i, high_i = int(round(low)), int(round(high))
        current_i = int(np.clip(round(current), low_i, high_i))
        if len(unique_values) <= 15:
            options = [int(v) for v in unique_values]
            return st.selectbox(
                f"{column} · 코드값", options,
                index=options.index(current_i) if current_i in options else 0,
                key=f"profile_{column}"
            )
        return st.slider(
            f"{column} · 값 조절", low_i, high_i, current_i, 1,
            key=f"profile_{column}"
        )

    step = max((high - low) / 100, 0.01)
    return st.slider(
        f"{column} · 값 조절", low, high, current, step,
        key=f"profile_{column}"
    )


with st.form("profile_form"):
    st.caption("선택한 학습 변수에 대해 값을 입력합니다. 입력 범위는 데이터셋의 관측 범위를 기준으로 합니다.")
    profile_values = {}
    profile_columns = st.columns(3)
    for index, column in enumerate(config["selected_features"]):
        with profile_columns[index % 3]:
            profile_values[column] = profile_input(
                column, profile_defaults.get(column, 0),
                data[column].dropna().unique().tolist()
            )
    profile_submitted = st.form_submit_button("가상 프로필 결과 계산", type="primary", use_container_width=True)

if profile_submitted:
    profile_df = pd.DataFrame([profile_values])
    result_cols = st.columns(len(trained_models))
    for col, (model_name, model) in zip(result_cols, trained_models.items()):
        scores = model.predict_proba(profile_df)[0]
        positive_index = list(model.classes_).index(1)
        positive_score = float(scores[positive_index])
        prediction = int(positive_score >= config["threshold"])
        with col:
            st.markdown(f"**{model_name}**")
            st.metric("모델의 양성 점수", f"{positive_score * 100:.1f}%")
            if prediction == 1:
                st.warning("분류 결과: 양성 범주 (1)")
            else:
                st.success("분류 결과: 음성 범주 (0)")
            st.caption(
                f"임계값: {config['threshold']:.2f}. 이 점수는 실제 질병 확률이나 진단 결과가 아닙니다."
            )

with st.expander("결과 해석 시 확인할 한계"):
    st.markdown(
        """
        1. 정확도만 보지 않고 민감도·특이도·위음성률을 함께 확인합니다.
        2. 집단별 평가 표본 수가 적으면 오류율이 불안정할 수 있습니다.
        3. 성별·연령 변수를 제외해도 다른 변수에 관련 정보가 반영될 수 있습니다.
        4. 집단별 성능 차이가 나타나도 그 자체로 차별의 원인이나 인과관계를 증명하지 않습니다.
        5. 미국 2015년 설문자료의 결과를 현재 한국 인구나 임상 환경에 그대로 일반화할 수 없습니다.
        """
    )

st.subheader("자료 출처")
st.markdown(
    f"- [UCI Machine Learning Repository — CDC Diabetes Health Indicators]({UCI_URL})\n"
    f"- [데이터 DOI / 인용 정보]({UCI_DOI})"
)
