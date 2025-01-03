import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import io

# Configure Streamlit page
st.set_page_config(
    page_title="Data Analysis & ML Models",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main { padding: 0rem 1rem; }
        .stButton>button {
            width: 100%;
            background-color: #ff4b4b;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.3rem;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #ff3333;
            transform: translateY(-2px);
        }
        h1, h2, h3 { color: #ff4b4b; }
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 2rem;
            background-color: #f0f2f6;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ff4b4b;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Model Classes
class LinearRegressionNumpy:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.w = None
        self.b = None
        self.losses = []
        
    def initialize_parameters(self):
        self.w = 0
        self.b = 0
        
    def forward(self, X):
        return X * self.w + self.b
    
    def compute_loss(self, y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)
    
    def train_step(self, X, y):
        y_pred = self.forward(X)
        m = len(X)
        dw = (2/m) * np.sum(X * (y_pred - y))
        db = (2/m) * np.sum(y_pred - y)
        
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db
        
        loss = self.compute_loss(y, y_pred)
        self.losses.append(loss)
        return loss
    
    def predict(self, X):
        return self.forward(X)

class LogisticRegressionNumpy:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.w = None
        self.b = None
        self.losses = []
    
    def initialize_parameters(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)
    
    def compute_loss(self, y_true, y_pred):
        m = len(y_true)
        loss = (-1/m) * np.sum(y_true * np.log(y_pred + 1e-15) + 
                              (1 - y_true) * np.log(1 - y_pred + 1e-15))
        return loss
    
    def train_step(self, X, y):
        m = len(y)
        y_pred = self.forward(X)
        
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db
        
        loss = self.compute_loss(y, y_pred)
        self.losses.append(loss)
        return loss
    
    def predict(self, X, threshold=0.5):
        return (self.forward(X) >= threshold).astype(int)

# Load data
@st.cache_data
def load_advertising_data():
    return pd.read_csv('advertising.csv')

@st.cache_data
def load_social_network_data():
    # Load the data
    df = pd.read_csv('social_network.csv')
    
    # Clean and preprocess
    # 1. Remove duplicates
    df = df.drop_duplicates()
    
    # 2. Convert Gender to numeric
    le = LabelEncoder()
    df['Gender_Encoded'] = le.fit_transform(df['Gender'])
    
    # 3. Handle any missing values (if any)
    df = df.dropna()
    
    # 4. Add age groups
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 20, 30, 40, 50, 100],
                            labels=['<20', '20-30', '30-40', '40-50', '50+'])
    
    # 5. Add salary ranges (in thousands)
    df['Salary_Group'] = pd.cut(df['EstimatedSalary'], 
                               bins=[0, 30000, 60000, 90000, 120000, float('inf')],
                               labels=['<30K', '30K-60K', '60K-90K', '90K-120K', '120K+'])
    
    return df

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/statistics.png", width=100)
    st.title("📊 Navigation")
    st.markdown("---")
    dataset_choice = st.radio("Choose Dataset", 
                            ["📺 Advertising (Linear Regression)", 
                             "🌐 Social Network (Logistic Regression)"])

# Main content
st.title("📊 Data Analysis & Machine Learning Models")
st.markdown("---")

if "📺 Advertising" in dataset_choice:
    data = load_advertising_data()
    
    tabs = st.tabs(["📋 Data Overview", "📈 EDA", "🎯 Linear Regression"])
    
    with tabs[0]:
        st.header("📋 Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Average TV Budget", f"${data['TV'].mean():,.2f}")
        with col3:
            st.metric("Average Sales", f"${data['Sales'].mean():,.2f}")
        
        st.dataframe(data.head(10), use_container_width=True)
        st.markdown("### 📊 Summary Statistics")
        st.dataframe(data.describe(), use_container_width=True)
    
    with tabs[1]:
        st.header("📈 Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            fig_tv = px.histogram(data, x='TV', 
                                title='TV Advertising Budget Distribution',
                                template="plotly_white")
            st.plotly_chart(fig_tv, use_container_width=True)
            
        with col2:
            fig_sales = px.histogram(data, x='Sales', 
                                   title='Sales Distribution',
                                   template="plotly_white")
            st.plotly_chart(fig_sales, use_container_width=True)
        
        fig_scatter = px.scatter(data, x='TV', y='Sales',
                               title='TV Advertising vs Sales',
                               template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tabs[2]:
        st.header("🎯 Linear Regression Model")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        with col2:
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
        with col3:
            n_epochs = st.number_input("Number of Epochs", 10, 1000, 100)
        
        if st.button("🚀 Train Model"):
            X = data['TV'].values
            y = data['Sales'].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).flatten()
            X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).flatten()
            
            model = LinearRegressionNumpy(learning_rate=learning_rate)
            model.initialize_parameters()
            
            col1, col2 = st.columns(2)
            loss_plot = col1.empty()
            pred_plot = col2.empty()
            metrics = st.empty()
            
            for epoch in range(n_epochs):
                loss = model.train_step(X_train_scaled, y_train)
                
                if epoch % 5 == 0:
                    with col1:
                        # Enhanced loss plot
                        fig_loss = go.Figure()
                        fig_loss.add_trace(
                            go.Scatter(
                                y=model.losses,
                                mode='lines',
                                name='Loss',
                                line=dict(color='#ff4b4b', width=2),
                                fill='tozeroy',
                                fillcolor='rgba(255,75,75,0.1)'
                            )
                        )
                        fig_loss.update_layout(
                            title={
                                'text': 'Training Progress',
                                'y':0.95,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'
                            },
                            xaxis_title='Epoch',
                            yaxis_title='Loss',
                            template="plotly_white",
                            hovermode='x unified',
                            showlegend=False
                        )
                        loss_plot.plotly_chart(fig_loss, use_container_width=True)
                    
                    with col2:
                        # Enhanced prediction plot
                        fig_pred = go.Figure()
                        
                        # Add training points
                        fig_pred.add_trace(
                            go.Scatter(
                                x=X_train,
                                y=y_train, 
                                mode='markers',
                                name='Training Data',
                                marker=dict(
                                    size=8,
                                    color='#2E86C1',
                                    symbol='circle',
                                    line=dict(color='white', width=1)
                                )
                            )
                        )
                        
                        # Add test points if available
                        fig_pred.add_trace(
                            go.Scatter(
                                x=X_test,
                                y=y_test,
                                mode='markers',
                                name='Test Data',
                                marker=dict(
                                    size=8,
                                    color='#28B463',
                                    symbol='diamond',
                                    line=dict(color='white', width=1)
                                )
                            )
                        )
                        
                        # Add regression line
                        X_line = np.linspace(X_train.min(), X_train.max(), 100)
                        X_line_scaled = scaler.transform(X_line.reshape(-1, 1)).flatten()
                        y_line = model.predict(X_line_scaled)
                        
                        fig_pred.add_trace(
                            go.Scatter(
                                x=X_line,
                                y=y_line,
                                mode='lines',
                                name='Regression Line',
                                line=dict(color='#ff4b4b', width=2)
                            )
                        )
                        
                        fig_pred.update_layout(
                            title={
                                'text': 'Model Predictions',
                                'y':0.95,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'
                            },
                            xaxis_title='TV Advertising Budget ($)',
                            yaxis_title='Sales ($)',
                            template="plotly_white",
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01,
                                bgcolor='rgba(255,255,255,0.8)'
                            ),
                            hovermode='closest'
                        )
                        pred_plot.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Calculate metrics
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                    train_mse = np.mean((y_pred_train - y_train) ** 2)
                    test_mse = np.mean((y_pred_test - y_test) ** 2)
                    
                    # Enhanced metrics display
                    metrics.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
                        <h3 style='color: #ff4b4b; margin-bottom: 15px;'>📊 Training Metrics</h3>
                        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;'>
                            <div>
                                <p style='color: #666; margin-bottom: 5px;'>Epoch</p>
                                <h4 style='margin: 0;'>{epoch + 1}</h4>
                            </div>
                            <div>
                                <p style='color: #666; margin-bottom: 5px;'>Training MSE</p>
                                <h4 style='margin: 0;'>{train_mse:.4f}</h4>
                            </div>
                            <div>
                                <p style='color: #666; margin-bottom: 5px;'>Test MSE</p>
                                <h4 style='margin: 0;'>{test_mse:.4f}</h4>
                            </div>
                            <div>
                                <p style='color: #666; margin-bottom: 5px;'>Model Parameters</p>
                                <h4 style='margin: 0;'>w: {model.w:.4f}, b: {model.b:.4f}</h4>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
else:  # Social Network Analysis
    data = load_social_network_data()
    
    tabs = st.tabs(["📋 Data Overview", "📈 EDA", "🎯 Logistic Regression"])
    
    with tabs[0]:
        st.header("📋 Data Overview")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Purchase Rate", f"{(data['Purchased'].mean()*100):.1f}%")
        with col3:
            st.metric("Average Age", f"{data['Age'].mean():.1f}")
        with col4:
            st.metric("Average Salary", f"${data['EstimatedSalary'].mean():,.0f}")
        
        # Gender distribution
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Gender Distribution")
            gender_dist = data['Gender'].value_counts()
            fig_gender_dist = px.pie(values=gender_dist.values, 
                                   names=gender_dist.index,
                                   title="Gender Distribution",
                                   color_discrete_sequence=['#ff4b4b', '#2E86C1'])
            st.plotly_chart(fig_gender_dist, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Purchase Distribution")
            purchase_dist = data['Purchased'].value_counts()
            fig_purchase_dist = px.pie(values=purchase_dist.values,
                                     names=['Not Purchased', 'Purchased'],
                                     title="Purchase Distribution",
                                     color_discrete_sequence=['#2E86C1', '#ff4b4b'])
            st.plotly_chart(fig_purchase_dist, use_container_width=True)
        
        # Data tables
        st.markdown("### 📋 Sample Data")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Summary statistics by group
        st.markdown("### 📊 Summary Statistics")
        
        tab1, tab2, tab3 = st.tabs(["Overall Stats", "Stats by Gender", "Stats by Purchase"])
        
        with tab1:
            st.dataframe(data[['Age', 'EstimatedSalary']].describe(), use_container_width=True)
        
        with tab2:
            st.dataframe(data.groupby('Gender')[['Age', 'EstimatedSalary']].describe(), use_container_width=True)
        
        with tab3:
            st.dataframe(data.groupby('Purchased')[['Age', 'EstimatedSalary']].describe(), use_container_width=True)
    
    with tabs[1]:
        st.header("📈 Exploratory Data Analysis")
        
        # Age and Salary Distribution
        col1, col2 = st.columns(2)
        with col1:
            fig_age = px.histogram(data, x='Age', 
                                 color='Purchased',
                                 title='Age Distribution by Purchase',
                                 template="plotly_white",
                                 color_discrete_sequence=['#2E86C1', '#ff4b4b'],
                                 marginal="box")
            st.plotly_chart(fig_age, use_container_width=True)
            
            # Age group analysis
            age_purchase_rates = data.groupby('Age_Group')['Purchased'].mean().reset_index()
            age_purchase_rates['Percentage'] = age_purchase_rates['Purchased'].map('{:.1%}'.format)
            fig_age_group = px.bar(age_purchase_rates,
                                 x='Age_Group',
                                 y='Purchased',
                                 title='Purchase Rate by Age Group',
                                 template="plotly_white",
                                 color_discrete_sequence=['#ff4b4b'])
            fig_age_group.update_traces(text=age_purchase_rates['Percentage'],
                                      textposition='outside')
            st.plotly_chart(fig_age_group, use_container_width=True)
            
        with col2:
            fig_salary = px.histogram(data, 
                                    x='EstimatedSalary',
                                    color='Purchased',
                                    title='Salary Distribution by Purchase',
                                    template="plotly_white",
                                    color_discrete_sequence=['#2E86C1', '#ff4b4b'],
                                    marginal="box")
            st.plotly_chart(fig_salary, use_container_width=True)
            
            # Salary group analysis
            salary_purchase_rates = data.groupby('Salary_Group')['Purchased'].mean().reset_index()
            salary_purchase_rates['Percentage'] = salary_purchase_rates['Purchased'].map('{:.1%}'.format)
            fig_salary_group = px.bar(salary_purchase_rates,
                                    x='Salary_Group',
                                    y='Purchased',
                                    title='Purchase Rate by Salary Group',
                                    template="plotly_white",
                                    color_discrete_sequence=['#ff4b4b'])
            fig_salary_group.update_traces(text=salary_purchase_rates['Percentage'],
                                         textposition='outside')
            st.plotly_chart(fig_salary_group, use_container_width=True)
        
        # Scatter plot with both categorical variables
        st.markdown("### 🎯 Age vs Salary Analysis")
        
        fig_scatter = px.scatter(data, 
                               x='Age',
                               y='EstimatedSalary',
                               color='Purchased',
                               symbol='Gender',
                               title='Age vs Salary by Purchase and Gender',
                               template="plotly_white",
                               color_discrete_sequence=['#2E86C1', '#ff4b4b'])
        
        fig_scatter.update_traces(marker=dict(size=10))
        fig_scatter.update_layout(
            xaxis_title="Age",
            yaxis_title="Estimated Salary ($)",
            legend_title="Purchase Status"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### 📊 Correlation Analysis")
        corr_data = data[['Age', 'EstimatedSalary', 'Purchased', 'Gender_Encoded']].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_data,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale=[[0, '#2E86C1'], [1, '#ff4b4b']],
            text=np.round(corr_data, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False))
        
        fig_corr.update_layout(
            title="Correlation Heatmap",
            template="plotly_white"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tabs[2]:
        st.header("🎯 Logistic Regression Model")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        with col2:
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
        with col3:
            n_epochs = st.number_input("Number of Epochs", 10, 1000, 100)
        
        if st.button("🚀 Train Model"):
            # Prepare features
            X = data[['Age', 'EstimatedSalary']].values
            y = data['Purchased'].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LogisticRegressionNumpy(learning_rate=learning_rate)
            model.initialize_parameters(n_features=2)
            
            col1, col2 = st.columns(2)
            loss_plot = col1.empty()
            decision_plot = col2.empty()
            metrics = st.empty()
            
            for epoch in range(n_epochs):
                loss = model.train_step(X_train_scaled, y_train)
                
                if epoch % 5 == 0:
                    with col1:
                        # Enhanced loss plot for Logistic Regression
                        fig_loss = go.Figure()
                        fig_loss.add_trace(
                            go.Scatter(
                                y=model.losses,
                                mode='lines',
                                name='Loss',
                                line=dict(color='#ff4b4b', width=2),
                                fill='tozeroy',
                                fillcolor='rgba(255,75,75,0.1)'
                            )
                        )
                        fig_loss.update_layout(
                            title={
                                'text': 'Training Progress',
                                'y':0.95,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'
                            },
                            xaxis_title='Epoch',
                            yaxis_title='Loss',
                            template="plotly_white",
                            hovermode='x unified',
                            showlegend=False
                        )
                        loss_plot.plotly_chart(fig_loss, use_container_width=True)
                    
                    with col2:
                        # Enhanced decision boundary plot
                        x_min, x_max = X_train[:, 0].min() - 2, X_train[:, 0].max() + 2
                        y_min, y_max = X_train[:, 1].min() - 2000, X_train[:, 1].max() + 2000
                        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                           np.linspace(y_min, y_max, 100))
                        
                        mesh_points = np.c_[xx.ravel(), yy.ravel()]
                        mesh_points_scaled = scaler.transform(mesh_points)
                        Z = model.predict(mesh_points_scaled)
                        Z = Z.reshape(xx.shape)
                        
                        fig_decision = go.Figure()
                        
                        # Add contour with custom colorscale
                        fig_decision.add_trace(
                            go.Contour(
                                x=xx[0],
                                y=yy[:, 0],
                                z=Z,
                                colorscale=[[0, 'rgba(31,119,180,0.2)'], 
                                          [1, 'rgba(255,75,75,0.2)']],
                                showscale=False,
                                contours=dict(
                                    showlines=False,
                                    coloring='heatmap'
                                ),
                                name='Decision Boundary'
                            )
                        )
                        
                        # Add decision boundary line
                        fig_decision.add_trace(
                            go.Contour(
                                x=xx[0],
                                y=yy[:, 0],
                                z=Z,
                                showscale=False,
                                contours=dict(
                                    showlines=True,
                                    type='constraint',
                                    operation='=',
                                    value=0.5,
                                    coloring='lines'
                                ),
                                line=dict(
                                    color='#ff4b4b',
                                    width=2
                                ),
                                name='Decision Boundary'
                            )
                        )
                        
                        # Add scatter points for training data
                        colors = ['#1f77b4' if label == 0 else '#ff4b4b' for label in y_train]
                        fig_decision.add_trace(
                            go.Scatter(
                                x=X_train[:, 0],
                                y=X_train[:, 1],
                                mode='markers',
                                marker=dict(
                                    size=8,
                                    color=colors,
                                    symbol='circle',
                                    line=dict(color='white', width=1)
                                ),
                                name='Training Data',
                                customdata=np.stack((y_train, X_train[:, 1]), axis=1),
                                hovertemplate="Age: %{x}<br>Salary: $%{y:,.0f}<br>Purchased: %{customdata[0]}"
                            )
                        )
                        
                        fig_decision.update_layout(
                            title={
                                'text': 'Decision Boundary & Training Data',
                                'y':0.95,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'
                            },
                            xaxis_title='Age',
                            yaxis_title='Estimated Salary ($)',
                            template="plotly_white",
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01,
                                bgcolor='rgba(255,255,255,0.8)'
                            ),
                            hovermode='closest'
                        )
                        
                        decision_plot.plotly_chart(fig_decision, use_container_width=True)
                    
                    # Calculate metrics
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                    train_acc = np.mean(y_pred_train == y_train)
                    test_acc = np.mean(y_pred_test == y_test)
                    
                    # Enhanced metrics display
                    metrics.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
                        <h3 style='color: #ff4b4b; margin-bottom: 15px;'>📊 Training Metrics</h3>
                        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;'>
                            <div>
                                <p style='color: #666; margin-bottom: 5px;'>Epoch</p>
                                <h4 style='margin: 0;'>{epoch + 1}</h4>
                            </div>
                            <div>
                                <p style='color: #666; margin-bottom: 5px;'>Loss</p>
                                <h4 style='margin: 0;'>{loss:.4f}</h4>
                            </div>
                            <div>
                                <p style='color: #666; margin-bottom: 5px;'>Training Accuracy</p>
                                <h4 style='margin: 0;'>{train_acc:.2%}</h4>
                            </div>
                            <div>
                                <p style='color: #666; margin-bottom: 5px;'>Test Accuracy</p>
                                <h4 style='margin: 0;'>{test_acc:.2%}</h4>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True) 