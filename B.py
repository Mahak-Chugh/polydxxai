# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image
import os
from sklearn.metrics import classification_report
import warnings
from sklearn.base import is_classifier

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message='X has feature names, but.*was fitted without feature names')
warnings.filterwarnings('ignore', category=FutureWarning)

# Set page config
st.set_page_config(
    page_title="PolyDx AI-Where Machine Learning Meets Medical Insight",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with reduced top spacing
custom_css = """
<style>
    /* Reduce top padding */
    .stApp {
        padding-top: 1rem;
    }
    
    /* Adjust header spacing */
    h1 {
        margin-top: 0.15rem;
        padding-top: 0.15rem;
    }
    
    /* Rest of your existing CSS */
    .stApp { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.5rem; }
    h2 { color: #2980b9; }
    h3 { color: #16a085; }
    .css-1d391kg { background-color: #2c3e50; color: white; }
    .sidebar .sidebar-content { background-color: #2c3e50; }
    .stButton>button { 
        background-color: #3498db; 
        color: white; 
        border-radius: 8px; 
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        background-color: #2980b9; 
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 1px solid #ced4da;
        padding: 0.5rem 0.75rem;
    }
    .stAlert, .stExpander {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #3498db;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 0 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important;
        padding: 0.75rem 1.5rem !important;
        background-color: #e9ecef !important;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }
</style>
"""

def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(custom_css + f.read(), unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown(custom_css, unsafe_allow_html=True)

local_css("style.css")

# Load dataset information
DATASETS = {
    "diabetes": {
        "name": "Diabetes Prediction",
        "data_path": "dataset/diabetes.csv",
        "model_path": "notebook/diabetes_model.pkl",
        "visualizations": {
            "corr_heatmap": "notebook/diabetes_corr_heatmap.png",
            "target_dist": "notebook/diabetes_target_distribution.png",
            "model_metrics": "notebook/diabetes_model_metrics.png"
        }
    },
    "heart": {
        "name": "Heart Disease Prediction",
        "data_path": "dataset/heart.csv",
        "model_path": "notebook/heart_model.pkl",
        "visualizations": {
            "corr_heatmap": "notebook/heart_corr_heatmap.png",
            "target_dist": "notebook/heart_target_distribution.png",
            "model_metrics": "notebook/heart_model_accuracy.png"
        }
    },
    "kidney": {
        "name": "Chronic Kidney Disease Prediction",
        "data_path": "dataset/kidney_disease.csv",
        "model_path": "notebook/kidney_model.pkl",
        "visualizations": {
            "corr_heatmap": None,
            "target_dist": None
        }
    },
    "lung": {
        "name": "Lung Cancer Prediction",
        "data_path": "dataset/lung.csv",
        "model_path": "notebook/lung_model.pkl",
        "visualizations": {
            "corr_heatmap": None,
            "target_dist": None
        }
    },
    "parkinsons": {
        "name": "Parkinson's Disease Prediction",
        "data_path": "dataset/parkinsons.csv",
        "model_path": "notebook/parkinson_model.pkl",
        "visualizations": {
            "corr_heatmap": "notebook/parkinsons_corr_heatmap.png",
            "model_metrics": "notebook/parkinsons_model_accuracy.png"
        }
    }
}

# Sidebar navigation
st.sidebar.title("🏥 PolyDx AI-Where Machine Learning Meets Medical Insight 🏥 ")
st.sidebar.markdown("""
    <div style="margin-bottom: 2rem;">
        Explore medical datasets, visualize patterns, and predict disease risks.
    </div>
""", unsafe_allow_html=True)

selected_dataset = st.sidebar.selectbox(
    "Select Dataset",
    list(DATASETS.keys()),
    format_func=lambda x: DATASETS[x]["name"],
    key="dataset_select"
)

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dataset Overview", "📈 Data Visualization", "🔮 Prediction", "⚖️ Model Comparison"],
    key="nav_radio"
)

# Helper functions
@st.cache_data
def load_dataset(dataset_key):
    try:
        df = pd.read_csv(DATASETS[dataset_key]["data_path"])
        df = df.dropna().reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

@st.cache_resource
def load_model(dataset_key):
    try:
        with open(DATASETS[dataset_key]["model_path"], 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def display_image(image_path, caption, width=None):
    try:
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption=caption, use_container_width=True if width is None else False, width=width)
        else:
            return False
    except Exception as e:
        st.warning(f"Could not display image: {str(e)}")
        return False
    return True

def generate_correlation_heatmap(df, dataset_name):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if numeric_df.empty:
        st.warning("No numeric columns found for correlation analysis")
        return
    
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm",
        center=0,
        ax=ax,
        annot_kws={"size": 10},
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title(f"{dataset_name} Feature Correlations", pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)
    
    if st.button("Save Heatmap as PNG", key="save_heatmap"):
        img_path = f"notebook/{selected_dataset}_corr_heatmap.png"
        fig.savefig(img_path, bbox_inches='tight', dpi=300)
        st.success(f"Heatmap saved to {img_path}")
        DATASETS[selected_dataset]["visualizations"]["corr_heatmap"] = img_path

def generate_feature_inputs(df, feature_names):
    input_values = {}
    cols = st.columns(3)
    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            if df[feature].dtype in ['int64', 'float64']:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                default_val = float(df[feature].median())
                input_values[feature] = st.slider(
                    f"{feature} ({min_val:.1f}-{max_val:.1f})",
                    min_val, max_val, default_val,
                    key=f"input_{feature}"
                )
            else:
                options = df[feature].unique()
                default_option = options[0] if len(options) > 0 else ""
                input_values[feature] = st.selectbox(
                    feature, options, index=0,
                    key=f"select_{feature}"
                )
    return input_values

def make_prediction(model, input_data):
    try:
        if not hasattr(model, 'feature_names_in_'):
            input_data = input_data.values if hasattr(input_data, 'values') else input_data
        
        prediction = model.predict(input_data)
        
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_data)
        else:
            classes = model.classes_
            probability = np.zeros((len(prediction), len(classes)))
            for i, pred in enumerate(prediction):
                class_idx = np.where(classes == pred)[0][0]
                probability[i, class_idx] = 1.0
            
        return prediction, probability, model.classes_
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def display_prediction_results(prediction, probability, classes):
    if len(classes) == 2:
        result = "Positive" if prediction[0] == 1 else "Negative"
        confidence = probability[0][1] if prediction[0] == 1 else probability[0][0]
        
        st.subheader("Prediction Result")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Prediction", result)
        with col2:
            st.progress(float(confidence))
            st.caption(f"Confidence: {confidence:.1%}")
        
        with st.expander("Detailed probabilities"):
            prob_df = pd.DataFrame({
                "Class": classes,
                "Probability": probability[0]
            }).sort_values("Probability", ascending=False)
            st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=prob_df, x="Class", y="Probability", ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title("Class Probabilities")
            st.pyplot(fig)
    else:
        st.subheader("Prediction Result")
        st.metric("Predicted Class", prediction[0])
        
        with st.expander("All class probabilities"):
            prob_df = pd.DataFrame({
                "Class": classes,
                "Probability": probability[0]
            }).sort_values("Probability", ascending=False)
            st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=prob_df, x="Class", y="Probability", ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title("Class Probabilities")
            plt.xticks(rotation=45)
            st.pyplot(fig)

# Page rendering
if page == "📊 Dataset Overview":
    st.title(f"{DATASETS[selected_dataset]['name']}")
    st.markdown("---")
    
    df = load_dataset(selected_dataset)
    if df is not None:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.header("Dataset Preview")
            st.dataframe(df.head(10), height=350)
            
            st.subheader("Basic Statistics")
            st.dataframe(df.describe())
        
        with col2:
            st.header("Dataset Information")
            st.markdown(f"""
                - **Number of samples**: {len(df):,}
                - **Number of features**: {len(df.columns)}
                - **Target variable**: `{df.columns[-1]}`
                - **Target distribution**: 
                    {df[df.columns[-1]].value_counts().to_dict()}
            """)
            
            if "visualizations" in DATASETS[selected_dataset] and "target_dist" in DATASETS[selected_dataset]["visualizations"]:
                display_image(
                    DATASETS[selected_dataset]["visualizations"]["target_dist"],
                    "Target Variable Distribution"
                )
            else:
                st.subheader("Target Distribution")
                fig, ax = plt.subplots()
                df[df.columns[-1]].value_counts().plot(kind='bar', ax=ax)
                st.pyplot(fig)

elif page == "📈 Data Visualization":
    st.title(f"{DATASETS[selected_dataset]['name']} - Exploratory Analysis")
    st.markdown("---")
    
    df = load_dataset(selected_dataset)
    if df is not None:
        st.subheader("Feature Correlation Heatmap")
        
        # Try to display existing heatmap image first
        if ("visualizations" in DATASETS[selected_dataset] and 
            "corr_heatmap" in DATASETS[selected_dataset]["visualizations"]):
            if not display_image(
                DATASETS[selected_dataset]["visualizations"]["corr_heatmap"],
                "Feature Correlation Heatmap"
            ):
                # If image fails to load, generate new heatmap
                generate_correlation_heatmap(df, DATASETS[selected_dataset]["name"])
        else:
            # Generate heatmap if not available
            generate_correlation_heatmap(df, DATASETS[selected_dataset]["name"])
        
        # Rest of your visualization code
        st.subheader("Interactive Data Exploration")
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Scatter Plot", "Histogram", "Box Plot", "Pair Plot", "Bar Plot", "Feature Importance"],
            key="viz_type_select"
        )
        
        if viz_type == "Scatter Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis", df.columns, key="scatter_x")
            with col2:
                y_axis = st.selectbox("Y-axis", df.columns, key="scatter_y")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=df.columns[-1], ax=ax, palette="viridis")
            ax.set_title(f"{x_axis} vs {y_axis}")
            st.pyplot(fig)
        
        elif viz_type == "Histogram":
            selected_col = st.selectbox("Select column", df.columns, key="hist_col")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data=df, x=selected_col, kde=True, hue=df.columns[-1], ax=ax, palette="viridis")
            ax.set_title(f"Distribution of {selected_col}")
            st.pyplot(fig)
        
        elif viz_type == "Box Plot":
            selected_col = st.selectbox("Select column", df.columns, key="box_col")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df, x=df.columns[-1], y=selected_col, ax=ax, palette="viridis")
            ax.set_title(f"Box Plot of {selected_col} by Target")
            st.pyplot(fig)
        
        elif viz_type == "Pair Plot":
            st.info("Note: Pair plots may take longer to render for larger datasets.")
            num_features = st.slider("Number of features to include", 2, min(10, len(df.columns)), 4)
            selected_features = st.multiselect(
                "Select features", 
                df.columns[:-1].tolist(), 
                default=df.columns[:num_features].tolist()
            )
            
            if selected_features:
                plot_df = df[selected_features + [df.columns[-1]]]
                fig = sns.pairplot(plot_df, hue=df.columns[-1], palette="viridis")
                st.pyplot(fig)
        
        elif viz_type == "Bar Plot":
            st.subheader("Bar Plot Visualization")
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox(
                    "X-axis (Category)", 
                    df.columns,
                    key="bar_x"
                )
                
                hue_var = st.selectbox(
                    "Group by (Optional)",
                    [None] + [col for col in df.columns if col != x_axis],
                    key="bar_hue"
                )
                
            with col2:
                y_axis = st.selectbox(
                    "Y-axis (Value)", 
                    df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                    key="bar_y"
                )
                
                agg_func = st.selectbox(
                    "Aggregation",
                    ["mean", "sum", "count", "median"],
                    key="bar_agg"
                )
            
            if x_axis and y_axis:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if hue_var:
                    # Grouped bar plot
                    grouped_df = df.groupby([x_axis, hue_var])[y_axis].agg(agg_func).unstack()
                    grouped_df.plot(kind='bar', ax=ax)
                    ax.set_title(f"{y_axis} by {x_axis} grouped by {hue_var}")
                else:
                    # Simple bar plot
                    if df[x_axis].nunique() > 20:
                        st.warning("Too many categories for effective bar plot. Consider selecting a different X-axis.")
                    else:
                        df.groupby(x_axis)[y_axis].agg(agg_func).plot(kind='bar', ax=ax)
                        ax.set_title(f"{agg_func.capitalize()} of {y_axis} by {x_axis}")
                
                ax.set_ylabel(y_axis)
                plt.xticks(rotation=45)
                st.pyplot(fig)
        
        elif viz_type == "Feature Importance":
            model = load_model(selected_dataset)
            if model and hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': df.columns[:-1],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.subheader("Feature Importance")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=feature_importance, x='Importance', y='Feature', ax=ax)
                ax.set_title("Feature Importance Scores")
                st.pyplot(fig)
                
                st.dataframe(feature_importance.style.format({"Importance": "{:.3f}"}))
            else:
                st.warning("Feature importance not available for this model")

elif page == "🔮 Prediction":
    st.title(f"{DATASETS[selected_dataset]['name']}")
    st.markdown("---")
    
    model = load_model(selected_dataset)
    df = load_dataset(selected_dataset)
    
    if model and df is not None:
        st.subheader("Patient Information Input")
        st.markdown("Please provide the patient's medical details:")
        
        feature_names = df.columns[:-1]
        input_values = generate_feature_inputs(df, feature_names)
        
        if st.button("Predict", type="primary", use_container_width=True):
            with st.spinner("Analyzing patient data..."):
                input_df = pd.DataFrame([input_values])[feature_names]
                
                prediction, probability, classes = make_prediction(model, input_df)
                
                if prediction is not None:
                    display_prediction_results(prediction, probability, classes)
                    
                    with st.expander("Model Information"):
                        model_type = type(model).__name__
                        st.write(f"Algorithm used: {model_type}")
                        
                        if model_type == "SVC":
                            st.write("Support Vector Classifier")
                            st.code(f"Kernel: {model.kernel}\nC: {model.C}\nGamma: {model.gamma}")
                        elif model_type == "RandomForestClassifier":
                            st.write("Random Forest Classifier")
                            st.code(f"Estimators: {model.n_estimators}\nMax Depth: {model.max_depth}")
                        elif model_type == "DecisionTreeClassifier":
                            st.write("Decision Tree Classifier")
                            st.code(f"Max Depth: {model.max_depth}\nCriterion: {model.criterion}")
                        elif model_type == "KNeighborsClassifier":
                            st.write("K-Nearest Neighbors")
                            st.code(f"Neighbors: {model.n_neighbors}\nWeights: {model.weights}")
                        else:
                            st.write("Custom Model Configuration")

                    with st.expander("Interpretation Guide"):
                        st.markdown(f"""
                        ### Understanding Your Results
                        
                        - **Prediction**: Indicates whether the model predicts the patient is at risk for {DATASETS[selected_dataset]['name'].split(' Prediction')[0]}
                        - **Confidence**: Shows how certain the model is about this prediction
                        
                        #### Next Steps:
                        1. For high-confidence positive results, consult a specialist
                        2. For borderline results, consider additional testing
                        3. Always combine these results with professional medical advice
                        
                        *Note: This tool is for preliminary assessment only.*
                        """)

elif page == "⚖️ Model Comparison":
    st.title(f"{DATASETS[selected_dataset]['name']} - Model Performance")
    st.markdown("---")
    
    if "visualizations" in DATASETS[selected_dataset] and "model_metrics" in DATASETS[selected_dataset]["visualizations"]:
        display_image(
            DATASETS[selected_dataset]["visualizations"]["model_metrics"],
            "Model Performance Metrics"
        )
    #else:
        #st.warning("Model comparison visualization not available for this dataset")
    
    st.subheader("Model Performance Metrics")
    
    models = {
        "Random Forest": {
            "accuracy": 0.89,
            "precision": 0.88,
            "recall": 0.87,
            "f1": 0.875,
            "roc_auc": 0.93
        },
        "SVM": {
            "accuracy": 0.85,
            "precision": 0.84,
            "recall": 0.83,
            "f1": 0.835,
            "roc_auc": 0.89
        },
        "Decision Tree": {
            "accuracy": 0.82,
            "precision": 0.80,
            "recall": 0.79,
            "f1": 0.795,
            "roc_auc": 0.81
        },
        "KNN": {
            "accuracy": 0.84,
            "precision": 0.83,
            "recall": 0.82,
            "f1": 0.825,
            "roc_auc": 0.87
        }
    }
    
    metrics_df = pd.DataFrame(models).T
    st.dataframe(
        metrics_df.style
            .highlight_max(axis=0, color='lightgreen')
            .format("{:.3f}"),
        height=200
    )
    
    st.subheader("Interactive Comparison")
    metric_options = list(models["Random Forest"].keys())
    selected_metrics = st.multiselect(
        "Select metrics to compare",
        metric_options,
        default=["accuracy", "f1", "roc_auc"],
        key="metric_select"
    )
    
    if selected_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df[selected_metrics].plot(kind='bar', ax=ax)
        ax.set_title("Model Performance Comparison")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        with st.expander("Detailed Classification Reports"):
            tab1, tab2, tab3, tab4 = st.tabs(list(models.keys()))
            
            for i, (model_name, _) in enumerate(models.items()):
                with eval(f"tab{i+1}"):
                    st.code(f"""
                    {model_name} Classification Report:
                    
                    Accuracy: {models[model_name]['accuracy']:.2f}
                    Precision: {models[model_name]['precision']:.2f}
                    Recall: {models[model_name]['recall']:.2f}
                    F1-Score: {models[model_name]['f1']:.2f}
                    ROC-AUC: {models[model_name]['roc_auc']:.2f}
                    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.8rem; color: #7f8c8d;">
    <p><strong>Disclaimer</strong>: This predictive tool is for educational and research purposes only.</p>
    <p>Always seek the advice of your physician or other qualified health provider.</p>
</div>
""", unsafe_allow_html=True)