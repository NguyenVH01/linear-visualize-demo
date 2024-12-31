import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io

# Configure Streamlit page and meta info
st.set_page_config(
    page_title="TV Advertising Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/hoang-nguyen-vu',
        'Report a bug': "https://github.com/hoang-nguyen-vu/issues",
        'About': """
        ## TV Advertising Analysis App
        
        This application analyzes the relationship between TV advertising budgets and sales using Simple Linear Regression.
        
        ### Features:
        - Interactive Data Exploration
        - Detailed Statistical Analysis
        - Real-time Model Training
        
        ### Tech Stack:
        - Python 3.8+
        - Streamlit
        - Pandas
        - NumPy
        - Plotly
        - Scikit-learn
        
        ### Meta Information:
        - Author: Hoang-Nguyen Vu
        - Version: 1.0.0
        - Description: Analyze the relationship between TV advertising budgets and sales
        - Keywords: TV Advertising, Sales Analysis, Linear Regression, Data Science
        - Language: English
        - Theme Color: #ff4b4b
        - Application Type: Web Analytics Tool
        - License: MIT
        """
    }
)


# Custom CSS with additional styling
st.markdown("""
    <style>
        .main {
            padding: 0rem 1rem;
        }
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
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .css-1d391kg {
            padding: 1rem;
        }
        h1 {
            color: #ff4b4b;
            padding-bottom: 2rem;
            text-align: center;
        }
        h2 {
            color: #ff4b4b;
            padding-bottom: 1rem;
        }
        h3 {
            color: #ff4b4b;
            padding-bottom: 0.5rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 2rem;
            background-color: #f0f2f6;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ff4b4b;
            color: white;
            transform: translateY(-2px);
        }
        /* Card hover effects */
        div[data-testid="stHorizontalBlock"] > div {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        div[data-testid="stHorizontalBlock"] > div:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f0f2f6;
            border-radius: 0.5rem;
        }
        .streamlit-expanderHeader:hover {
            background-color: #e8eaed;
        }
        /* Footer styling */
        footer {
            visibility: hidden;
        }
        /* Custom footer */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f0f2f6;
            padding: 1rem;
            text-align: center;
            font-size: 0.8rem;
        }
    </style>
    
    <!-- Add custom footer -->
    <div class="footer">
        Made with ❤️ by Hoang-Nguyen Vu | © 2024 All rights reserved
    </div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('advertising.csv')

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.w = None  # weight
        self.b = None  # bias
        self.losses = []  # Store losses for plotting
        
    def initialize_parameters(self):
        self.w = 0
        self.b = 0
        
    def forward(self, X):
        return X * self.w + self.b
    
    def compute_loss(self, y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)
    
    def train_step(self, X, y):
        # Forward pass
        y_pred = self.forward(X)
        
        # Compute gradients
        m = len(X)
        dw = (2/m) * np.sum(X * (y_pred - y))
        db = (2/m) * np.sum(y_pred - y)
        
        # Update parameters
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db
        
        # Compute loss
        loss = self.compute_loss(y, y_pred)
        self.losses.append(loss)
        
        return loss
    
    def predict(self, X):
        return self.forward(X)

# Load data
data = load_data()

# Prepare data for model training
X = data['TV'].values
y = data['Sales'].values

# Sidebar styling and content
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/statistics.png", width=100)
    st.title("📊 Navigation")
    st.markdown("---")
    page = st.radio("", ["📋 Data Overview", "📈 EDA", "🎯 Model Training"])
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application analyzes the relationship between TV advertising budgets and sales.
    
    **Features:**
    - Data Overview
    - Exploratory Data Analysis
    - Linear Regression Model
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ by Hoang-Nguyen Vu")

# Main content
st.title("📺 TV Advertising Impact Analysis")
st.markdown("---")

if "📋 Data Overview" in page:
    st.header("📋 Data Overview")
    
    # Data summary in cards using columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div style='padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;'>
            <h3 style='margin: 0; color: #ff4b4b;'>Total Records</h3>
            <p style='font-size: 2rem; margin: 0;'>{}</p>
        </div>
        """.format(len(data)), unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;'>
            <h3 style='margin: 0; color: #ff4b4b;'>Avg TV Budget</h3>
            <p style='font-size: 2rem; margin: 0;'>${:,.2f}</p>
        </div>
        """.format(data['TV'].mean()), unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;'>
            <h3 style='margin: 0; color: #ff4b4b;'>Avg Sales</h3>
            <p style='font-size: 2rem; margin: 0;'>${:,.2f}</p>
        </div>
        """.format(data['Sales'].mean()), unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div style='padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;'>
            <h3 style='margin: 0; color: #ff4b4b;'>Correlation</h3>
            <p style='font-size: 2rem; margin: 0;'>{:.2f}</p>
        </div>
        """.format(data['TV'].corr(data['Sales'])), unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📊 Raw Data", "📈 Statistics", "ℹ️ Info"])
    
    with tab1:
        st.dataframe(data.head(10), use_container_width=True)
        
    with tab2:
        st.dataframe(data.describe(), use_container_width=True)
        
    with tab3:
        buffer = io.StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())

elif "📈 EDA" in page:
    st.header("📈 Exploratory Data Analysis")
    
    tabs = st.tabs(["📊 Distribution", "🔄 Relationship", "📑 Statistical Tests"])
    
    with tabs[0]:
        st.subheader("Distribution Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig_tv_dist = px.histogram(data, x='TV', 
                                     title='Distribution of TV Advertising Budget',
                                     labels={'TV': 'TV Advertising Budget ($)', 'count': 'Frequency'},
                                     template="plotly_white")
            fig_tv_dist.add_trace(px.box(data, x='TV').data[0])
            fig_tv_dist.update_layout(
                showlegend=False,
                title_x=0.5,
                title_font_size=20
            )
            st.plotly_chart(fig_tv_dist, use_container_width=True)
            
            with st.expander("📊 TV Advertising Statistics"):
                st.write(f"**Mean:** ${data['TV'].mean():,.2f}")
                st.write(f"**Median:** ${data['TV'].median():,.2f}")
                st.write(f"**Std Dev:** ${data['TV'].std():,.2f}")
            
        with col2:
            fig_sales_dist = px.histogram(data, x='Sales', 
                                        title='Distribution of Sales',
                                        labels={'Sales': 'Sales ($)', 'count': 'Frequency'},
                                        template="plotly_white")
            fig_sales_dist.add_trace(px.box(data, x='Sales').data[0])
            fig_sales_dist.update_layout(
                showlegend=False,
                title_x=0.5,
                title_font_size=20
            )
            st.plotly_chart(fig_sales_dist, use_container_width=True)
            
            with st.expander("📊 Sales Statistics"):
                st.write(f"**Mean:** ${data['Sales'].mean():,.2f}")
                st.write(f"**Median:** ${data['Sales'].median():,.2f}")
                st.write(f"**Std Dev:** ${data['Sales'].std():,.2f}")
    
    with tabs[1]:
        st.subheader("Relationship Analysis")
        
        fig_scatter = px.scatter(data, x='TV', y='Sales',
                               trendline="ols",
                               title='TV Advertising vs Sales Relationship',
                               labels={'TV': 'TV Advertising Budget ($)', 'Sales': 'Sales ($)'},
                               template="plotly_white")
        fig_scatter.update_layout(
            title_x=0.5,
            title_font_size=20
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        with st.expander("📊 Correlation Analysis"):
            correlation = data['TV'].corr(data['Sales'])
            st.write(f"**Correlation coefficient:** {correlation:.4f}")
            st.write("""
            **Interpretation:**
            - Strong positive correlation (> 0.7)
            - Indicates that higher TV advertising budgets are associated with higher sales
            """)
    
    with tabs[2]:
        st.subheader("Statistical Tests")
        
        from scipy import stats
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style='padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;'>
                <h4 style='color: #ff4b4b;'>TV Advertising - Normality Test</h4>
            """, unsafe_allow_html=True)
            
            stat_tv, p_tv = stats.shapiro(data['TV'])
            st.write(f"**Statistic:** {stat_tv:.4f}")
            st.write(f"**p-value:** {p_tv:.4f}")
            st.write("**Result:** " + ("Normal" if p_tv > 0.05 else "Not normal"))
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style='padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;'>
                <h4 style='color: #ff4b4b;'>Sales - Normality Test</h4>
            """, unsafe_allow_html=True)
            
            stat_sales, p_sales = stats.shapiro(data['Sales'])
            st.write(f"**Statistic:** {stat_sales:.4f}")
            st.write(f"**p-value:** {p_sales:.4f}")
            st.write("**Result:** " + ("Normal" if p_sales > 0.05 else "Not normal"))
            st.markdown("</div>", unsafe_allow_html=True)

else:  # Model Training
    st.header("🎯 Model Training")
    
    # Model parameters in an expander
    with st.expander("⚙️ Model Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05,
                                help="Proportion of dataset to include in the test split")
        with col2:
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f",
                                          help="Step size at each iteration while moving toward a minimum of the loss function")
        with col3:
            n_epochs = st.number_input("Number of Epochs", 10, 1000, 100,
                                     help="Number of complete passes through the training dataset")
    
    # Split and scale data when parameters change
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).flatten()
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).flatten()
    
    # Train model button with better styling
    if st.button("🚀 Train Model"):
        with st.spinner('Training in progress...'):
            # Initialize model
            model = SimpleLinearRegression(learning_rate=learning_rate)
            model.initialize_parameters()
            
            # Create placeholders for plots using columns
            col1, col2 = st.columns(2)
            with col1:
                loss_plot = st.empty()
            with col2:
                prediction_plot = st.empty()
            metrics = st.empty()
            
            # Training loop
            for epoch in range(n_epochs):
                loss = model.train_step(X_train_scaled, y_train)
                
                if epoch % 5 == 0:  # Update plots every 5 epochs
                    # Plot loss in first column
                    with col1:
                        fig_loss = px.line(y=model.losses, 
                                         title='Training Progress',
                                         labels={'x': 'Epoch', 'y': 'Loss'},
                                         template="plotly_white")
                        fig_loss.update_layout(title_x=0.5, title_font_size=20)
                        loss_plot.plotly_chart(fig_loss, use_container_width=True)
                    
                    # Plot predictions in second column
                    with col2:
                        y_pred_train = model.predict(X_train_scaled)
                        fig_pred = px.scatter(template="plotly_white")
                        fig_pred.add_scatter(x=X_train, 
                                           y=y_train, 
                                           name='Training Data',
                                           mode='markers')
                        X_line = np.linspace(X_train.min(), X_train.max(), 100)
                        X_line_scaled = scaler.transform(X_line.reshape(-1, 1)).flatten()
                        y_line = model.predict(X_line_scaled)
                        fig_pred.add_scatter(x=X_line,
                                           y=y_line,
                                           name='Regression Line',
                                           mode='lines')
                        fig_pred.update_layout(
                            title='Real-time Predictions',
                            title_x=0.5,
                            title_font_size=20,
                            xaxis_title='TV Advertising Budget ($)',
                            yaxis_title='Sales ($)'
                        )
                        prediction_plot.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Display metrics in a styled container
                    y_pred_test = model.predict(X_test_scaled)
                    train_mse = np.mean((y_pred_train - y_train) ** 2)
                    test_mse = np.mean((y_pred_test - y_test) ** 2)
                    metrics.markdown(f"""
                    <div style='padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;'>
                        <h4 style='color: #ff4b4b; margin-bottom: 0.5rem;'>Training Metrics</h4>
                        <p><strong>Epoch:</strong> {epoch + 1}</p>
                        <p><strong>Training MSE:</strong> {train_mse:.4f}</p>
                        <p><strong>Test MSE:</strong> {test_mse:.4f}</p>
                        <p><strong>Weight (w):</strong> {model.w:.4f}</p>
                        <p><strong>Bias (b):</strong> {model.b:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.success('Training completed!')
            
            # Final results
            st.markdown("### 📊 Final Results")
            col1, col2 = st.columns(2)
            with col1:
                fig_loss = px.line(y=model.losses, 
                                 title='Final Training Progress',
                                 labels={'x': 'Epoch', 'y': 'Loss'},
                                 template="plotly_white")
                fig_loss.update_layout(title_x=0.5, title_font_size=20)
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col2:
                y_pred_test = model.predict(X_test_scaled)
                fig_final = px.scatter(template="plotly_white")
                fig_final.add_scatter(x=X_train, 
                                    y=y_train, 
                                    name='Training Data',
                                    mode='markers')
                fig_final.add_scatter(x=X_test,
                                    y=y_test,
                                    name='Test Data',
                                    mode='markers')
                X_line = np.linspace(X.min(), X.max(), 100)
                X_line_scaled = scaler.transform(X_line.reshape(-1, 1)).flatten()
                y_line = model.predict(X_line_scaled)
                fig_final.add_scatter(x=X_line,
                                    y=y_line,
                                    name='Regression Line',
                                    mode='lines')
                fig_final.update_layout(
                    title='Final Model Performance',
                    title_x=0.5,
                    title_font_size=20,
                    xaxis_title='TV Advertising Budget ($)',
                    yaxis_title='Sales ($)'
                )
                st.plotly_chart(fig_final, use_container_width=True) 