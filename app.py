import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. SYNTHETIC DATA GENERATION FOR PROJECT LIFECYCLE (UNCHANGED) ---
@st.cache_data(show_spinner="Generating high-fidelity synthetic project lifecycle data...")
def generate_synthetic_lifecycle_data(n_projects=100, stages_per_project=12, seed=42):
    """
    Generates synthetic data for complete power grid project lifecycles.
    Each project has multiple stages with material demand predictions.
    """
    np.random.seed(seed)
    random.seed(seed)

    # Define categorical features
    tower_types = ['Lattice-Suspension', 'Lattice-Tension', 'Tubular-Monopole', 'T-Pylon', 'Terminal']
    substation_types = ['765kV AIS', '400kV GIS', '220kV Conventional', '132kV Distribution']
    materials = ['Structural Steel (MT)', 'Aluminium Conductor (KM)', 'Insulator Sets (Units)', 
                 'Concrete (m3)', 'GIS Switchgear (Units)', 'Cable Trays (Units)', 'Control Systems (Units)']
    locations = ['North-East', 'South', 'West-Desert', 'Central-Plains', 'Himalayan-Region']
    seasons = ['Dry', 'Monsoon', 'Winter']
    project_stages = ['Planning & Design', 'Procurement', 'Fabrication', 'Construction Phase 1', 
                      'Construction Phase 2', 'Installation', 'Testing & Commissioning']

    data = []
    
    for proj_id in range(n_projects):
        project_type = np.random.choice(['Transmission Line', 'Substation', 'Distribution Network', 
                                        'Renewable Integration', 'Smart Grid Upgrade'])
        tower_type = np.random.choice(tower_types)
        substation_type = np.random.choice(substation_types)
        location = np.random.choice(locations)
        budget = np.random.uniform(500, 5000)
        tax_rate = np.random.uniform(5.0, 18.0)
        lead_time = np.random.randint(30, 270)
        
        # Generate demand across project lifecycle stages
        for stage_num, stage in enumerate(project_stages):
            for month in range(1, stages_per_project + 1):
                # Progress calculation based on stage
                progress = (stage_num / len(project_stages)) * 100 + (month / stages_per_project) * (100 / len(project_stages))
                progress = min(99, progress)
                
                # Material selection
                material = np.random.choice(materials)
                season = np.random.choice(seasons)
                
                # Base demand calculation with stage-specific multipliers
                stage_multipliers = {
                    'Planning & Design': 0.1,
                    'Procurement': 1.8,  # HIGHEST - main procurement phase
                    'Fabrication': 1.2,
                    'Construction Phase 1': 1.5,
                    'Construction Phase 2': 1.4,
                    'Installation': 0.9,
                    'Testing & Commissioning': 0.3
                }
                
                base_demand = (budget / 100) + (100 - progress) * 3
                stage_factor = stage_multipliers[stage]
                
                # Seasonal factor
                season_factor = {'Dry': 1.1, 'Monsoon': 0.6, 'Winter': 1.0}[season]
                
                # Location complexity
                location_factor = {'Himalayan-Region': 1.3, 'Central-Plains': 0.9, 'West-Desert': 1.2}.get(location, 1.0)
                
                # Material-specific multiplier
                material_multiplier = {
                    'Structural Steel (MT)': 1.5, 'Aluminium Conductor (KM)': 0.8, 
                    'Insulator Sets (Units)': 0.5, 'Concrete (m3)': 1.2, 
                    'GIS Switchgear (Units)': 2.0, 'Cable Trays (Units)': 1.1,
                    'Control Systems (Units)': 0.6
                }[material]
                
                # Calculate final demand
                demand = (base_demand * stage_factor * season_factor * location_factor * material_multiplier 
                         + np.random.normal(0, 30))
                demand = max(0, demand)
                
                data.append({
                    'Project_ID': f'P{proj_id:04d}',
                    'Project_Type': project_type,
                    'Stage': stage,
                    'Stage_Num': stage_num,
                    'Month_In_Stage': month,
                    'Overall_Progress_%': progress,
                    'Material_Name': material,
                    'Tower_Type': tower_type,
                    'Substation_Type': substation_type,
                    'Geographic_Location': location,
                    'Project_Budget_Cr': budget,
                    'Tax_Rate_%': tax_rate,
                    'Season': season,
                    'Lead_Time_Days': lead_time,
                    'Demand_Quantity': int(np.round(demand))
                })

    return pd.DataFrame(data)

# --- 2. PREPARE SEQUENCES FOR LSTM (UNCHANGED) ---
def prepare_lstm_sequences(df, seq_length=12):
    """Prepares time series sequences for LSTM training"""
    
    # Group by project and sort by stage
    sequences = []
    targets = []
    
    # Required features for LSTM input (MUST be present in uploaded data)
    required_features = ['Overall_Progress_%', 'Project_Budget_Cr', 'Tax_Rate_%', 'Lead_Time_Days']

    # Check if all required features are in the DataFrame
    if not all(feature in df.columns for feature in required_features):
        st.error(f"Missing required columns for LSTM training: {set(required_features) - set(df.columns)}")
        raise ValueError("Dataframe missing required features for sequence generation.")

    
    for project_id in df['Project_ID'].unique():
        # Ensure project data is sortable by stage and month
        project_data = df[df['Project_ID'] == project_id].sort_values(['Stage_Num', 'Month_In_Stage'], inplace=False)
        
        # Extract features for LSTM
        features = project_data[required_features].values
        targets_data = project_data['Demand_Quantity'].values
        
        # Create sequences
        for i in range(len(project_data) - seq_length):
            sequences.append(features[i:i+seq_length])
            targets.append(targets_data[i+seq_length])
    
    if not sequences:
        st.error("Not enough data to form sequences. Check sequence length or project size.")
        raise ValueError("Cannot form LSTM sequences.")

    return np.array(sequences), np.array(targets)

# --- 3. ENCODE CATEGORICAL FEATURES (UNCHANGED) ---
def encode_categorical_features(df):
    """Encodes categorical features for the model"""
    # NOTE: This function is present in the original code but not actually used in the LSTM flow,
    # as the LSTM only uses numerical features: Overall_Progress_%, Project_Budget_Cr, Tax_Rate_%, Lead_Time_Days.
    # We keep it for consistency or potential future use.
    df_encoded = df.copy()
    
    le_dict = {}
    categorical_cols = ['Material_Name', 'Tower_Type', 'Substation_Type', 'Geographic_Location', 'Season', 'Stage', 'Project_Type']
    
    # Filter for columns that actually exist in the dataframe (important for uploaded data)
    present_categorical_cols = [col for col in categorical_cols if col in df_encoded.columns]

    for col in present_categorical_cols:
        le = LabelEncoder()
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str)) # Convert to string for robustness
        le_dict[col] = le
    
    return df_encoded, le_dict

# --- 4. BUILD LSTM MODEL (UNCHANGED) ---
def build_lstm_nexus_model(seq_length, n_features):
    """
    Builds an advanced LSTM model optimized for power grid material demand forecasting.
    Predicts demand across ALL project lifecycle stages.
    """
    model = Sequential([
        # LSTM Layer 1 - Learn temporal patterns
        LSTM(128, activation='relu', input_shape=(seq_length, n_features), return_sequences=True),
        Dropout(0.2),
        
        # LSTM Layer 2 - Deeper temporal understanding
        LSTM(256, activation='relu', return_sequences=True),
        Dropout(0.2),
        
        # LSTM Layer 3 - Extract final patterns
        LSTM(128, activation='relu', return_sequences=False),
        Dropout(0.2),
        
        # Dense layers for final prediction
        Dense(64, activation='relu'),
        Dropout(0.1),
        
        Dense(32, activation='relu'),
        
        # Output layer (non-negative demand)
        Dense(1, activation='relu')
    ], name='NEXUS_LSTM_Demand_Forecaster')
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', 'mse'])
    return model

# --- 5. TRAIN LSTM MODEL (MODIFIED FOR CACHE CLEARANCE) ---
@st.cache_resource(show_spinner="Training LSTM-NEXUS model on project lifecycle data...")
def train_lstm_model(df, seq_length=12, epochs=100, batch_size=32):
    """Trains the LSTM model"""
    
    # Prepare sequences
    X_seq, y_seq = prepare_lstm_sequences(df, seq_length)
    
    # Normalize targets
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_seq.reshape(-1, 1)).flatten()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_scaled, test_size=0.2, random_state=42
    )
    
    # Build and train
    model = build_lstm_nexus_model(seq_length, X_train.shape[2])
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0
    )
    
    # Evaluate
    loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
    
    # Inverse transform predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_original = scaler_y.inverse_transform(y_pred).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'mae': mean_absolute_error(y_test_original, y_pred_original),
        'mse': mean_squared_error(y_test_original, y_pred_original),
        'rmse': np.sqrt(mean_squared_error(y_test_original, y_pred_original)),
        'r2': r2_score(y_test_original, y_pred_original)
    }
    
    return model, history.history, metrics, scaler_y

# --- 6. GENERATE FORECASTS FOR ALL STAGES (UNCHANGED) ---
def forecast_all_stages(df, model, scaler_y, seq_length=12):
    """Generates demand forecasts for every stage of the project"""
    
    forecasts = []
    
    for project_id in df['Project_ID'].unique():
        project_data = df[df['Project_ID'] == project_id].sort_values(['Stage_Num', 'Month_In_Stage'])
        
        for idx, row in project_data.iterrows():
            # Get context (past 12 months)
            # NOTE: The original code does not actually use the model.predict() here 
            # for the forecast, it just uses the 'Demand_Quantity' from the input DataFrame.
            # For a real application, you would use a sliding window approach with model.predict().
            
            # Calculate confidence based on stage progress
            stage_num = row['Stage_Num']
            progress = row['Overall_Progress_%']
            
            if stage_num == 1:  # Procurement stage
                confidence = 0.95
                priority = "🔴 CRITICAL"
            elif stage_num in [2, 3, 4]:  # Construction stages
                confidence = 0.88 + (progress / 100) * 0.07
                priority = "🟡 HIGH"
            elif stage_num in [5, 6]:  # Installation/Testing
                confidence = 0.85 + (progress / 100) * 0.10
                priority = "🟢 MEDIUM"
            else:
                confidence = 0.80
                priority = "🔵 LOW"
            
            # Generate recommendation
            demand_q = row['Demand_Quantity']
            if 'Demand_Quantity' in df.columns:
                q75 = df['Demand_Quantity'].quantile(0.75)
                q50 = df['Demand_Quantity'].quantile(0.50)
                if demand_q > q75:
                    recommendation = "Urgent procurement - initiate orders immediately"
                elif demand_q > q50:
                    recommendation = "Plan procurement - coordinate with vendors"
                else:
                    recommendation = "Monitor demand - routine procurement"
            else:
                recommendation = "Demand analysis pending"
            
            forecasts.append({
                'Project_ID': project_id,
                'Project_Type': row.get('Project_Type', 'N/A'),
                'Stage': row.get('Stage', 'N/A'),
                'Material': row.get('Material_Name', 'N/A'),
                'Location': row.get('Geographic_Location', 'N/A'),
                'Progress_%': round(progress, 1),
                'Predicted_Demand': int(demand_q), # Using actual demand as 'prediction' for now
                'Confidence_%': round(confidence * 100, 1),
                'Priority': priority,
                'Lead_Time_Days': row.get('Lead_Time_Days', 0),
                'Recommendation': recommendation,
                'Budget_Cr': round(row.get('Project_Budget_Cr', 0), 1)
            })
    
    return pd.DataFrame(forecasts)

# --- 7. PROJECT-LEVEL TESTING FUNCTION (UNCHANGED) ---
def generate_project_material_plan(df, project_id):
    """
    Generates detailed stage-wise material requirements for a single project.
    This is critical for supply chain planning, procurement scheduling, and inventory optimization.
    """
    # NOTE: This function assumes the full set of synthetic data columns are present.
    required_cols = ['Stage', 'Stage_Num', 'Material_Name', 'Demand_Quantity', 
                     'Lead_Time_Days', 'Project_Budget_Cr', 'Geographic_Location', 'Project_Type']
    
    # Filter for required columns
    project_data = df[df['Project_ID'] == project_id][required_cols].copy()
    
    if len(project_data) == 0:
        return None
    
    # Group by stage and material
    stage_material_plan = project_data.groupby(['Stage', 'Stage_Num', 'Material_Name']).agg({
        'Demand_Quantity': 'sum',
        'Lead_Time_Days': 'first',
        'Project_Budget_Cr': 'first',
        'Geographic_Location': 'first',
        'Project_Type': 'first'
    }).reset_index()
    
    # Sort by stage
    stage_material_plan = stage_material_plan.sort_values('Stage_Num')
    
    # Calculate cumulative demand
    stage_material_plan['Cumulative_Demand'] = stage_material_plan.groupby('Material_Name')['Demand_Quantity'].cumsum()
    
    # Add procurement timing recommendations
    stage_material_plan['Procurement_Start_Week'] = stage_material_plan.apply(
        lambda row: max(1, (row['Stage_Num'] * 12) - (row['Lead_Time_Days'] // 7)), axis=1
    )
    
    # Inventory buffer recommendations
    stage_material_plan['Safety_Stock_%'] = stage_material_plan['Stage_Num'].apply(
        lambda x: 20 if x <= 2 else (15 if x <= 4 else 10)
    )
    
    stage_material_plan['Recommended_Order_Qty'] = (
        stage_material_plan['Demand_Quantity'] * (1 + stage_material_plan['Safety_Stock_%'] / 100)
    ).astype(int)
    
    return stage_material_plan

# --- 8. HELPER FUNCTION TO VALIDATE UPLOADED DATA ---
def validate_and_transform_uploaded_data(uploaded_df):
    """
    Validates and transforms uploaded data to match the required synthetic data format 
    for training and forecasting.
    """
    required_cols = [
        'Project_ID', 'Project_Type', 'Stage', 'Stage_Num', 'Month_In_Stage',
        'Overall_Progress_%', 'Material_Name', 'Lead_Time_Days', 'Demand_Quantity',
        'Project_Budget_Cr', 'Tax_Rate_%', 'Geographic_Location'
    ]
    
    missing_cols = [col for col in required_cols if col not in uploaded_df.columns]
    
    if missing_cols:
        st.error(f"Uploaded file is missing critical columns: {', '.join(missing_cols)}. Please check the required format.")
        return None
        
    # Ensure dtypes are correct for training/sorting
    try:
        df = uploaded_df[required_cols].copy()
        df['Stage_Num'] = df['Stage_Num'].astype(int)
        df['Month_In_Stage'] = df['Month_In_Stage'].astype(int)
        df['Overall_Progress_%'] = df['Overall_Progress_%'].astype(float)
        df['Demand_Quantity'] = df['Demand_Quantity'].astype(int)
        df['Project_Budget_Cr'] = df['Project_Budget_Cr'].astype(float)
        df['Tax_Rate_%'] = df['Tax_Rate_%'].astype(float)
        df['Lead_Time_Days'] = df['Lead_Time_Days'].astype(int)
        return df
    except Exception as e:
        st.error(f"Data type conversion failed. Ensure numerical columns are correctly formatted. Error: {e}")
        return None

# --- 9. STREAMLIT APP (MODIFIED) ---
def streamlit_app():
    st.set_page_config(layout="wide", page_title="NEXUS-LSTM: Infrastructure Material Demand Forecasting")
    
    st.title(" 📈 NEXUS-LSTM: Smart Material Demand Forecasting")
    st.markdown("**Advanced LSTM Neural Network for Power Grid Project Lifecycle Demand Prediction**")

    # Session State
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
        st.session_state.model_trained = False
        st.session_state.data = None
        st.session_state.model = None
        st.session_state.metrics = None
        st.session_state.forecasts = None
        st.session_state.data_source = 'Synthetic'

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        st.markdown("---")
        
        st.subheader("🌐 Data Source Selection")
        data_source = st.radio(
            "Choose your data source:",
            ('Synthetic', 'Upload CSV/Excel'),
            key='data_source_radio'
        )
        
        # Clear state if source changes
        if st.session_state.data_source != data_source:
            st.session_state.data_generated = False
            st.session_state.model_trained = False
            st.session_state.data = None
            st.session_state.model = None
            st.session_state.metrics = None
            st.session_state.forecasts = None
            st.session_state.data_source = data_source
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()


        if data_source == 'Synthetic':
            st.markdown("---")
            st.subheader("📊 Synthetic Data Configuration")
            n_projects = st.slider("Number of Projects", 50, 300, 100)
            stages_per_project = st.slider("Months per Stage", 6, 24, 12)
            random_seed = st.number_input("Random Seed", value=42)
            
            if st.button("🔄 Generate Project Lifecycle Data", use_container_width=True):
                with st.spinner("Generating synthetic lifecycle data..."):
                    st.session_state.data = generate_synthetic_lifecycle_data(n_projects, stages_per_project, random_seed)
                    st.session_state.data_generated = True
                    st.session_state.model_trained = False
                st.success(f"✅ Generated {len(st.session_state.data)} lifecycle records")
        
        elif data_source == 'Upload CSV/Excel':
            st.markdown("---")
            st.subheader("📁 Upload Data File")
            uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
            
            if uploaded_file is not None:
                try:
                    # Read file
                    if uploaded_file.name.endswith('.csv'):
                        uploaded_df = pd.read_csv(uploaded_file)
                    else:
                        uploaded_df = pd.read_excel(uploaded_file)
                        
                    # Validate and transform
                    processed_df = validate_and_transform_uploaded_data(uploaded_df)
                    
                    if processed_df is not None:
                        st.session_state.data = processed_df
                        st.session_state.data_generated = True
                        st.session_state.model_trained = False
                        st.success(f"✅ Uploaded and validated {len(st.session_state.data)} records")
                        st.markdown("---")
                        st.markdown("💡 **Required Columns:** `Project_ID`, `Stage`, `Stage_Num`, `Month_In_Stage`, `Overall_Progress_%`, `Demand_Quantity`, `Project_Budget_Cr`, `Tax_Rate_%`, `Lead_Time_Days`, `Material_Name`, `Project_Type`, `Geographic_Location`")
                    else:
                        st.session_state.data_generated = False
                        st.session_state.model_trained = False
                except Exception as e:
                    st.error(f"Error reading or processing file: {e}")
                    st.session_state.data_generated = False
                    st.session_state.model_trained = False
        
        # --- Model Training Section (Only if data is ready) ---
        st.markdown("---")
        st.subheader("🤖 Model Training")
        
        if st.session_state.data_generated:
            epochs = st.slider("Training Epochs", 50, 200, 100, key='epochs')
            batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32, key='batch_size')
            seq_length = st.slider("Sequence Length (months)", 6, 24, 12, key='seq_length')
            
            if st.button("🚀 Train LSTM-NEXUS Model", use_container_width=True):
                with st.spinner("Training LSTM model on project lifecycle..."):
                    try:
                        # Clear resource cache before training a new model/data combination
                        st.cache_resource.clear() 
                        
                        model, history, metrics, scaler_y = train_lstm_model(
                            st.session_state.data, seq_length, epochs, batch_size
                        )
                        st.session_state.model = model
                        st.session_state.metrics = metrics
                        st.session_state.scaler_y = scaler_y
                        st.session_state.model_trained = True
                        
                        # Generate forecasts
                        st.session_state.forecasts = forecast_all_stages(
                            st.session_state.data, model, scaler_y, seq_length
                        )
                    except ValueError as ve:
                        # Catch error from prepare_lstm_sequences if features are missing or data is insufficient
                        st.error(f"Model Training Error: {ve}")
                        st.session_state.model_trained = False
                    except Exception as e:
                        st.error(f"An unexpected error occurred during training: {e}")
                        st.session_state.model_trained = False
                
                if st.session_state.model_trained:
                    st.success("✅ LSTM Model trained successfully!")
        
        st.markdown("---")
        st.subheader("📈 Model Status")
        
        if st.session_state.model_trained:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE", f"{st.session_state.metrics['mae']:.2f}")
                st.metric("RMSE", f"{st.session_state.metrics['rmse']:.2f}")
            with col2:
                st.metric("MSE", f"{st.session_state.metrics['mse']:.2f}")
                st.metric("R² Score", f"{st.session_state.metrics['r2']:.4f}")
            
            st.success("✅ Model Ready for Forecasting")
        elif st.session_state.data_generated:
            st.warning("⚠️ Data ready. Train model to proceed.")
        else:
            st.info("💡 Select a data source and process data first.")

    # Main Content
    if not st.session_state.data_generated:
        st.info("👈 Use the sidebar to generate or upload project lifecycle data and train the LSTM model.")
        st.markdown("""
        ### 🎯 About NEXUS-LSTM
        - **Technology**: Advanced LSTM (Long Short-Term Memory) neural network
        - **Capability**: Predicts material demand across ALL project lifecycle stages
        - **Stages Covered**: Planning, Procurement, Fabrication, Construction, Installation, Testing
        - **Smart Features**: Stage-specific forecasting, confidence scoring, priority alerts
        - **Supply Chain Focus**: Procurement planning, inventory optimization, lead time management
        """)
        return

    # Dashboard
    st.header("📊 Project Lifecycle Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Projects", st.session_state.data['Project_ID'].nunique())
    with col2:
        st.metric("Total Records", len(st.session_state.data))
    with col3:
        # Safely count unique stages
        stages = st.session_state.data.get('Stage', pd.Series([]))
        st.metric("Lifecycle Stages", stages.nunique() if not stages.empty else 0)
    with col4:
        # Safely count unique materials
        materials = st.session_state.data.get('Material_Name', pd.Series([]))
        st.metric("Material Types", materials.nunique() if not materials.empty else 0)
    
    st.dataframe(st.session_state.data.head(15), use_container_width=True)

    if st.session_state.model_trained:
        st.header("🎯 Demand Forecasts Across Project Lifecycle")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 All Forecasts", "🔬 Project Testing", "📈 Analytics", "💾 Export", "🤖 Model Info"])
        
        # NOTE: Subsequent tabs (1-5) rely heavily on the structure/content 
        # of the synthetic data generation and the forecast_all_stages output. 
        # They are kept largely UNCHANGED for now, assuming uploaded data 
        # adheres to the required column structure.
        
        with tab1:
            st.subheader("Complete Lifecycle Demand Forecasts")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                # Use .get() to safely access keys for uploaded data
                stages = st.session_state.forecasts.get('Stage', pd.Series([]))
                selected_stage = st.multiselect("Filter by Stage", 
                                                stages.unique().tolist(),
                                                default=stages.unique().tolist())
            with col2:
                priority = st.session_state.forecasts.get('Priority', pd.Series([]))
                selected_priority = st.multiselect("Filter by Priority",
                                                   priority.unique().tolist(),
                                                   default=priority.unique().tolist())
            with col3:
                location = st.session_state.forecasts.get('Location', pd.Series([]))
                selected_location = st.multiselect("Filter by Location",
                                                   location.unique().tolist(),
                                                   default=location.unique().tolist())
            
            # Apply filters
            if not st.session_state.forecasts.empty:
                filtered = st.session_state.forecasts[
                    (st.session_state.forecasts['Stage'].isin(selected_stage)) &
                    (st.session_state.forecasts['Priority'].isin(selected_priority)) &
                    (st.session_state.forecasts['Location'].isin(selected_location))
                ]
                st.dataframe(filtered, use_container_width=True, height=400)
            else:
                st.info("No forecasts to display.")
        
        with tab2:
            st.subheader("🔬 Project-Level Material Requirement Testing")
            st.markdown("**Supply Chain Planning & Procurement Optimization Dashboard**")
            
            # Mode selection
            test_mode = st.radio(
                "Select Testing Mode",
                ["📋 Existing Project", "🎛️ Custom Parameters (Manual Input)"],
                horizontal=True,
                key='test_mode'
            )
            
            if test_mode == "📋 Existing Project":
                # Project selector
                available_projects = sorted(st.session_state.data['Project_ID'].unique())
                selected_test_project = st.selectbox(
                    "Select Project for Detailed Material Plan",
                    available_projects,
                    index=0,
                    key='selected_project'
                )
                
                if selected_test_project:
                    # Get project parameters (safely handle missing columns for uploaded data)
                    project_info = st.session_state.data[st.session_state.data['Project_ID'] == selected_test_project].iloc[0]
                    project_type = project_info.get('Project_Type', 'N/A')
                    location = project_info.get('Geographic_Location', 'N/A')
                    budget = project_info.get('Project_Budget_Cr', 0.0)
                    lead_time = project_info.get('Lead_Time_Days', 0)
                else:
                    st.warning("No projects available.")
                    return
                
            else:
                # Custom parameter inputs (reusing logic from original script)
                st.markdown("### 🎛️ Configure Project Parameters")
                st.markdown("Adjust these parameters to see predicted material demand at each development stage")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    project_type = st.selectbox(
                        "Project Type",
                        ['Transmission Line', 'Substation', 'Distribution Network', 
                         'Renewable Integration', 'Smart Grid Upgrade'],
                         key='custom_type'
                    )
                
                with col2:
                    location = st.selectbox(
                        "Location",
                        ['North-East', 'South', 'West-Desert', 'Central-Plains', 'Himalayan-Region'],
                        key='custom_location'
                    )
                
                with col3:
                    budget = st.number_input(
                        "Budget (Cr ₹)",
                        min_value=500.0,
                        max_value=5000.0,
                        value=2454.8,
                        step=100.0,
                        key='custom_budget'
                    )
                
                with col4:
                    lead_time = st.number_input(
                        "Avg Lead Time (Days)",
                        min_value=30,
                        max_value=270,
                        value=126,
                        step=5,
                        key='custom_lead_time'
                    )
                
                selected_test_project = "CUSTOM_PROJECT"
            
            if selected_test_project:
                # Display current parameters
                st.markdown("---")
                st.markdown("### 📊 Current Project Configuration")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Project Type", project_type)
                with col2:
                    st.metric("Location", location)
                with col3:
                    st.metric("Budget (Cr)", f"₹{budget:.1f}")
                with col4:
                    st.metric("Avg Lead Time", f"{lead_time} days")
                
                st.markdown("---")
                
                # Generate material demand predictions based on parameters
                if test_mode == "🎛️ Custom Parameters (Manual Input)":
                    # --- Custom Project Prediction Logic (from original script) ---
                    st.markdown("### 🎯 Predicted Material Demand by Development Stage")
                    st.markdown("**Based on your configured parameters**")
                    
                    project_stages = ['Planning & Design', 'Procurement', 'Fabrication', 'Construction Phase 1', 
                                      'Construction Phase 2', 'Installation', 'Testing & Commissioning']
                    materials = ['Structural Steel (MT)', 'Aluminium Conductor (KM)', 'Insulator Sets (Units)', 
                                 'Concrete (m3)', 'GIS Switchgear (Units)', 'Cable Trays (Units)', 'Control Systems (Units)']
                    stage_multipliers = {
                        'Planning & Design': 0.1, 'Procurement': 1.8, 'Fabrication': 1.2,
                        'Construction Phase 1': 1.5, 'Construction Phase 2': 1.4,
                        'Installation': 0.9, 'Testing & Commissioning': 0.3
                    }
                    location_factors = {
                        'Himalayan-Region': 1.3, 'Central-Plains': 0.9, 'West-Desert': 1.2,
                        'North-East': 1.1, 'South': 1.0
                    }
                    material_multipliers = {
                        'Structural Steel (MT)': 1.5, 'Aluminium Conductor (KM)': 0.8, 
                        'Insulator Sets (Units)': 0.5, 'Concrete (m3)': 1.2, 
                        'GIS Switchgear (Units)': 2.0, 'Cable Trays (Units)': 1.1,
                        'Control Systems (Units)': 0.6
                    }
                    
                    predictions = []
                    for stage_num, stage in enumerate(project_stages):
                        stage_factor = stage_multipliers[stage]
                        location_factor = location_factors[location]
                        
                        for material in materials:
                            material_factor = material_multipliers[material]
                            base_demand = (budget / 100) * stage_factor * location_factor * material_factor
                            demand = max(0, base_demand + np.random.normal(0, 10))
                            safety_stock_pct = 20 if stage_num <= 2 else (15 if stage_num <= 4 else 10)
                            recommended_qty = int(demand * (1 + safety_stock_pct / 100))
                            procurement_start_week = max(1, (stage_num * 12) - (lead_time // 7))
                            
                            predictions.append({
                                'Stage': stage, 'Stage_Num': stage_num, 'Material': material, 
                                'Required_Qty': int(demand), 'Safety_Stock_%': safety_stock_pct, 
                                'Recommended_Order_Qty': recommended_qty, 
                                'Procurement_Start_Week': procurement_start_week, 'Lead_Time_Days': lead_time
                            })
                    
                    material_plan = pd.DataFrame(predictions)
                    
                else:
                    # Use existing project data
                    material_plan = generate_project_material_plan(st.session_state.data, selected_test_project)
                
                if material_plan is not None:
                    
                    # Stage-wise material breakdown
                    st.subheader("📋 Stage-wise Material Requirements")
                    st.markdown("**For Supply Chain Planning, Procurement Scheduling & Inventory Management**")
                    
                    # Display material plan
                    if test_mode == "🎛️ Custom Parameters (Manual Input)":
                        display_plan = material_plan[[
                            'Stage', 'Material', 'Required_Qty', 'Recommended_Order_Qty',
                            'Safety_Stock_%', 'Procurement_Start_Week', 'Lead_Time_Days'
                        ]].copy()
                        display_plan.columns = [
                            'Stage', 'Material', 'Required Qty', 'Order Qty (with buffer)',
                            'Safety Stock %', 'Start Procurement (Week)', 'Lead Time (Days)'
                        ]
                    else:
                        display_plan = material_plan[[
                            'Stage', 'Material_Name', 'Demand_Quantity', 'Recommended_Order_Qty',
                            'Safety_Stock_%', 'Procurement_Start_Week', 'Lead_Time_Days'
                        ]].copy()
                        display_plan.columns = [
                            'Stage', 'Material', 'Required Qty', 'Order Qty (with buffer)',
                            'Safety Stock %', 'Start Procurement (Week)', 'Lead Time (Days)'
                        ]
                    
                    st.dataframe(display_plan, use_container_width=True, height=400)
                    
                    # Summary statistics & Charts (unchanged)
                    st.markdown("---")
                    st.subheader("📊 Material Summary & Procurement Insights")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Total Material Requirements by Stage**")
                        qty_col = 'Required_Qty' if test_mode == "🎛️ Custom Parameters (Manual Input)" else 'Demand_Quantity'
                        stage_totals = material_plan.groupby('Stage')[qty_col].sum().sort_values(ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        stage_totals.plot(kind='barh', ax=ax, color='steelblue')
                        ax.set_xlabel('Total Demand Quantity')
                        ax.set_title(f'Material Demand by Project Stage\n{project_type} - {location}')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("**Material-wise Total Demand**")
                        material_col = 'Material' if test_mode == "🎛️ Custom Parameters (Manual Input)" else 'Material_Name'
                        material_totals = material_plan.groupby(material_col)[qty_col].sum().sort_values(ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        material_totals.plot(kind='barh', ax=ax, color='coral')
                        ax.set_xlabel('Total Demand Quantity')
                        ax.set_title(f'Total Demand by Material Type\nBudget: ₹{budget:.1f} Cr')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Procurement timeline (unchanged logic)
                    st.markdown("---")
                    st.subheader("📅 Procurement Timeline & Planning")
                    
                    if test_mode == "🎛️ Custom Parameters (Manual Input)":
                        timeline_data = material_plan[['Stage', 'Material', 'Procurement_Start_Week', 'Lead_Time_Days', 'Required_Qty']].copy()
                    else:
                        timeline_data = material_plan[['Stage', 'Material_Name', 'Procurement_Start_Week', 'Lead_Time_Days', 'Demand_Quantity']].copy()
                        timeline_data.columns = ['Stage', 'Material', 'Procurement_Start_Week', 'Lead_Time_Days', 'Required_Qty']
                    
                    timeline_data['Procurement_End_Week'] = timeline_data['Procurement_Start_Week'] + (timeline_data['Lead_Time_Days'] // 7)
                    
                    fig, ax = plt.subplots(figsize=(14, 8))
                    
                    for idx, row in timeline_data.iterrows():
                        ax.barh(
                            f"{row['Material']} ({row['Stage']})", 
                            row['Lead_Time_Days'] // 7,
                            left=row['Procurement_Start_Week'],
                            alpha=0.7
                        )
                    
                    ax.set_xlabel('Project Week')
                    ax.set_title(f'Material Procurement Timeline (Gantt Chart)\nLead Time: {lead_time} days')
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Key insights (unchanged logic)
                    st.markdown("---")
                    st.subheader("💡 Key Procurement Insights")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total_demand = material_plan[qty_col].sum()
                        st.metric("Total Material Demand", f"{int(total_demand):,} units")
                    
                    with col2:
                        peak_stage = stage_totals.index[0]
                        st.metric("Peak Demand Stage", peak_stage)
                    
                    with col3:
                        critical_material = material_totals.index[0]
                        st.metric("Highest Demand Material", critical_material.split('(')[0].strip())
                    
                    # Export project-specific plan (unchanged logic)
                    st.markdown("---")
                    st.subheader("💾 Export Project Material Plan")
                    
                    csv_buffer = material_plan.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"📥 Download {selected_test_project} Material Plan (CSV)",
                        data=csv_buffer,
                        file_name=f'{selected_test_project}_material_plan.csv',
                        mime='text/csv',
                    )
                else:
                    st.warning("No data available for selected project")

        # Tab 3, 4, 5 logic remains largely the same, relying on st.session_state.forecasts
        with tab3:
            # Analytics (safe access)
            if not st.session_state.forecasts.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Demand by Project Stage")
                    stage_demand = st.session_state.forecasts.groupby('Stage')['Predicted_Demand'].sum().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    stage_demand.plot(kind='barh', ax=ax, color='steelblue')
                    ax.set_xlabel('Total Predicted Demand')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Demand by Material")
                    material_demand = st.session_state.forecasts.groupby('Material')['Predicted_Demand'].sum().sort_values(ascending=False).head(10)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    material_demand.plot(kind='barh', ax=ax, color='coral')
                    ax.set_xlabel('Total Predicted Demand')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                st.subheader("Demand Trend Across Project Lifecycle")
                demand_trend = st.session_state.forecasts.groupby(['Stage'])['Predicted_Demand'].agg(['sum', 'mean', 'std']).reset_index()
                
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.bar(demand_trend['Stage'], demand_trend['sum'], alpha=0.7, label='Total Demand', color='blue')
                ax.set_xlabel('Project Stage')
                ax.set_ylabel('Predicted Demand')
                ax.set_title('Material Demand Distribution Across Project Lifecycle')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.subheader("Priority Distribution")
                priority_dist = st.session_state.forecasts['Priority'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 5))
                priority_dist.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=['#FF6B6B', '#FFA500', '#4CAF50', '#2196F3'])
                st.pyplot(fig)
            else:
                st.info("No forecast data to generate analytics.")
        
        with tab4:
            # Export (safe access)
            st.subheader("📥 Export Forecast Results")
            
            if not st.session_state.forecasts.empty and st.session_state.data is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_buffer = st.session_state.forecasts.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download All Forecasts (CSV)",
                        data=csv_buffer,
                        file_name='nexus_lifecycle_forecasts.csv',
                        mime='text/csv',
                    )
                
                with col2:
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        st.session_state.forecasts.to_excel(writer, index=False, sheet_name='Forecasts')
                        st.session_state.data.to_excel(writer, index=False, sheet_name='Raw Data')
                    
                    st.download_button(
                        label="📥 Download All Forecasts (Excel)",
                        data=excel_buffer.getvalue(),
                        file_name='nexus_lifecycle_forecasts.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    )
            else:
                st.info("No data to export.")
        
        with tab5:
            # Model Info (safe access)
            st.subheader("🤖 LSTM-NEXUS Model Architecture")
            
            if st.session_state.metrics:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **Model Architecture (LSTM)**
                    
                    - **LSTM Layer 1**: 128 units, ReLU, 20% Dropout
                    - **LSTM Layer 2**: 256 units, ReLU, 20% Dropout  
                    - **LSTM Layer 3**: 128 units, ReLU, 20% Dropout
                    - **Dense 1**: 64 units, ReLU, 10% Dropout
                    - **Dense 2**: 32 units, ReLU
                    - **Output**: 1 unit, ReLU (non-negative)
                    
                    **Training Config**
                    - Loss: MSE (Mean Squared Error)
                    - Optimizer: Adam (lr=0.001)
                    - Sequence Length: 12 months (or selected)
                    """)
                
                with col2:
                    st.markdown("""
                    **Performance Metrics**
                    """)
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("MAE", f"{st.session_state.metrics['mae']:.2f}")
                        st.metric("R² Score", f"{st.session_state.metrics['r2']:.4f}")
                    with col_m2:
                        st.metric("RMSE", f"{st.session_state.metrics['rmse']:.2f}")
                        st.metric("MSE", f"{st.session_state.metrics['mse']:.2f}")
                
                st.markdown("---")
            
            st.subheader("📈 Supply Chain Optimization Benefits")
            # ... (Unchanged static text)

# Execute the Streamlit application
if __name__ == "__main__":
    streamlit_app()