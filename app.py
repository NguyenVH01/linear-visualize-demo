import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="Simple Linear Regression", layout="wide")

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

st.title("Simple Linear Regression - TV Advertising vs Sales")

# Model parameters
st.sidebar.subheader("Model Parameters")
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
n_epochs = st.sidebar.number_input("Number of Epochs", 10, 1000, 100)

# Prepare data
X = data['TV'].values
y = data['Sales'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).flatten()
X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).flatten()

# Train model
if st.button("Train Model"):
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
                                 title='Loss over Epochs',
                                 labels={'x': 'Epoch', 'y': 'Loss'})
                loss_plot.plotly_chart(fig_loss, use_container_width=True)
            
            # Plot predictions in second column
            with col2:
                y_pred_train = model.predict(X_train_scaled)
                fig_pred = px.scatter()
                # Training data
                fig_pred.add_scatter(x=X_train, 
                                   y=y_train, 
                                   name='Training Data',
                                   mode='markers')
                # Prediction line
                X_line = np.linspace(X_train.min(), X_train.max(), 100)
                X_line_scaled = scaler.transform(X_line.reshape(-1, 1)).flatten()
                y_line = model.predict(X_line_scaled)
                fig_pred.add_scatter(x=X_line,
                                   y=y_line,
                                   name='Regression Line',
                                   mode='lines')
                fig_pred.update_layout(title='TV Advertising vs Sales',
                                     xaxis_title='TV Advertising Budget',
                                     yaxis_title='Sales')
                prediction_plot.plotly_chart(fig_pred, use_container_width=True)
            
            # Display metrics
            y_pred_test = model.predict(X_test_scaled)
            train_mse = np.mean((y_pred_train - y_train) ** 2)
            test_mse = np.mean((y_pred_test - y_test) ** 2)
            metrics.write(f"""
            **Current Metrics:**
            - Epoch: {epoch + 1}
            - Training MSE: {train_mse:.4f}
            - Test MSE: {test_mse:.4f}
            - Weight (w): {model.w:.4f}
            - Bias (b): {model.b:.4f}
            """)
    
    # Final predictions plot with test data
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Progress")
        fig_loss = px.line(y=model.losses, 
                          title='Final Loss over Epochs',
                          labels={'x': 'Epoch', 'y': 'Loss'})
        st.plotly_chart(fig_loss, use_container_width=True)
    
    with col2:
        st.subheader("Final Model Performance")
        y_pred_test = model.predict(X_test_scaled)
        fig_final = px.scatter()
        # Training data
        fig_final.add_scatter(x=X_train, 
                            y=y_train, 
                            name='Training Data',
                            mode='markers')
        # Test data
        fig_final.add_scatter(x=X_test,
                            y=y_test,
                            name='Test Data',
                            mode='markers')
        # Prediction line
        X_line = np.linspace(X.min(), X.max(), 100)
        X_line_scaled = scaler.transform(X_line.reshape(-1, 1)).flatten()
        y_line = model.predict(X_line_scaled)
        fig_final.add_scatter(x=X_line,
                            y=y_line,
                            name='Regression Line',
                            mode='lines')
        fig_final.update_layout(title='Final Model: TV Advertising vs Sales',
                              xaxis_title='TV Advertising Budget',
                              yaxis_title='Sales')
        st.plotly_chart(fig_final, use_container_width=True) 