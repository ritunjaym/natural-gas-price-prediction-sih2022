# Natural Gas Price Prediction Dashboard

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-FF6B6B?style=for-the-badge)](https://natural-gas-price-prediction-sih2022.streamlit.app/)
[![SIH 2022](https://img.shields.io/badge/🏆_SIH_2022-FFD93D?style=for-the-badge)](https://sih.gov.in/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)

> **🏆 Smart India Hackathon 2022 Project** - A comprehensive energy market forecasting solution developed for India's premier national innovation challenge

**🌐 [Experience Live Dashboard](https://natural-gas-price-prediction-sih2022.streamlit.app/)**

## 📊 Project Overview

An advanced time series forecasting application that predicts natural gas spot prices using SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) modeling. The system provides interactive visualizations and real-time predictions through a responsive Streamlit dashboard, enabling energy market analysts and traders to make data-driven decisions.

## 🏆 Smart India Hackathon 2022 Achievement

This project was developed for **Smart India Hackathon 2022**, India's largest nationwide innovation challenge organized by the Government of India. Among thousands of participants from premier institutions across the country, this solution was recognized for addressing critical energy sector challenges through advanced data science and machine learning techniques.

**Competition Highlights:**
- **National Recognition**: Selected from 15,000+ registrations across 3,000+ institutions
- **Problem Statement**: Energy price forecasting for market stability and policy planning
- **Innovation Impact**: Demonstrated practical application of statistical modeling for national energy security
- **Technical Excellence**: Implemented production-ready forecasting system with real-time capabilities

### 🎯 Business Value

- **Market Intelligence**: Provides 2-5 year natural gas price forecasts with confidence intervals
- **Risk Management**: Enables informed hedging strategies for energy portfolio management
- **Cost Planning**: Supports long-term budgeting for energy-intensive operations
- **Investment Decisions**: Assists in evaluating energy sector opportunities
- **Policy Support**: Aids government and regulatory bodies in energy market planning

### 🛠️ Technology Stack

**Backend & Analytics**
- **Python 3.8+**: Core application development
- **Pandas**: Data manipulation and time series analysis
- **Statsmodels**: SARIMAX implementation and statistical modeling
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Statistical plotting and seasonal decomposition

**Frontend & Visualization**
- **Streamlit**: Interactive web application framework
- **Plotly**: Dynamic, responsive charting and visualization
- **CSS**: Custom styling and responsive design

**Data Management**
- **CSV Processing**: Efficient data ingestion and export
- **Time Series Indexing**: Optimized temporal data handling

## 🔍 Methodology & Features

### Time Series Modeling
- **SARIMAX (1,1,1)(1,1,0,12)**: Captures both trend and seasonal patterns
- **Grid Search Optimization**: Systematic parameter tuning for optimal model performance
- **Seasonal Decomposition**: Separates trend, seasonal, and residual components
- **Rolling Forecasts**: Generates predictions with statistical confidence intervals

### Dashboard Capabilities

#### 📈 Predictive Analytics
- **Multi-year Forecasts**: 2021-2026 price projections
- **Confidence Intervals**: Upper and lower bounds for risk assessment
- **Interactive Time Series**: Zoom, pan, and explore historical patterns
- **Point-in-time Predictions**: Specific date price queries

#### 📊 Historical Analysis
- **Commodity Comparisons**: Multi-asset price correlation analysis
- **Consumption Patterns**: Natural gas usage breakdown by sector
- **Seasonal Decomposition**: Visual trend and seasonality analysis

#### 🎛️ User Experience
- **Responsive Design**: Optimized for desktop and mobile viewing
- **File Upload Support**: Custom dataset analysis capability
- **Export Functionality**: Download predictions in CSV format
- **Real-time Processing**: Dynamic model execution with progress indicators

## 📁 Data Sources & Processing

### Input Data Requirements
```csv
Date,Spot_price
1997-01-01,3.45
1997-02-01,2.15
...
```

### Supported Formats
- **Historical Spot Prices**: Monthly natural gas price data (1997-2022)
- **Production Data**: U.S. natural gas production statistics
- **Consumption Data**: Sectoral consumption breakdowns
- **Storage Data**: Underground storage capacity and utilization

## 🚀 Installation & Usage

### Prerequisites
```bash
Python 3.8+
pip package manager
```

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/natural-gas-price-prediction.git
cd natural-gas-price-prediction

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
```

### Requirements
```text
streamlit
pandas
plotly
numpy
statsmodels
matplotlib
```

## 📊 Model Performance & Results

### Statistical Metrics
- **Mean Absolute Error (MAE)**: Quantifies prediction accuracy
- **Root Mean Square Error (RMSE)**: Measures forecast reliability
- **Confidence Intervals**: 95% prediction bounds for risk assessment

### Key Insights
- **Seasonal Patterns**: Strong correlation with heating/cooling demand cycles
- **Market Volatility**: Captured extreme price movements (2008, 2021)
- **Long-term Trends**: Identified structural shifts in natural gas markets

### Forecast Accuracy
The model demonstrates robust performance in capturing:
- Short-term price movements (1-3 months)
- Seasonal volatility patterns
- Long-term market trends
- Economic shock responses

## 🎯 Use Cases

### Energy Trading
- **Portfolio Optimization**: Risk-adjusted position sizing
- **Hedging Strategies**: Forward contract pricing guidance
- **Market Timing**: Entry/exit signal generation

### Corporate Planning
- **Budget Forecasting**: Energy cost projections for financial planning
- **Supply Chain**: Natural gas procurement strategy optimization
- **Investment Analysis**: ROI calculations for energy infrastructure

### Research & Analysis
- **Market Studies**: Academic and institutional research support
- **Policy Analysis**: Regulatory impact assessment
- **Competitive Intelligence**: Market positioning and strategy development

## 🔮 Future Enhancements

### Advanced Modeling
- [ ] **Machine Learning Integration**: LSTM/GRU neural networks for complex pattern recognition
- [ ] **Multivariate Analysis**: Incorporate weather, economic indicators, and geopolitical factors
- [ ] **Ensemble Methods**: Combine multiple models for improved accuracy

### Technical Improvements
- [ ] **Real-time Data Integration**: Live price feeds and automatic model updating
- [ ] **API Development**: RESTful endpoints for programmatic access
- [ ] **Cloud Deployment**: Scalable infrastructure on AWS/Azure/GCP
- [ ] **Database Integration**: PostgreSQL/MongoDB for efficient data management

### User Experience
- [ ] **Advanced Filters**: Custom date ranges and scenario analysis
- [ ] **Alert System**: Price threshold notifications and trend alerts
- [ ] **Mobile App**: Native iOS/Android applications
- [ ] **User Authentication**: Personalized dashboards and saved preferences

## 📈 Project Impact & Recognition

### Smart India Hackathon 2022 Success
This application represents a **nationally recognized solution** to critical energy market challenges:

- **Innovation at Scale**: Competed against India's brightest minds from IITs, NITs, and premier institutions
- **Real-world Application**: Addressed genuine industry problems identified by government ministries
- **Technical Rigor**: Demonstrated advanced statistical modeling and full-stack development capabilities
- **Production Deployment**: Successfully deployed on cloud infrastructure for public access

### Business Impact Metrics
- **Reduces forecasting time** from weeks to minutes through automated modeling
- **Improves decision accuracy** with statistically validated predictions
- **Enables scenario planning** through interactive what-if analysis
- **Supports risk management** with confidence interval quantification
- **Serves national interest** by contributing to energy market transparency

## 🌐 Live Application

**Experience the dashboard**: [https://natural-gas-price-prediction-sih2022.streamlit.app/](https://natural-gas-price-prediction-sih2022.streamlit.app/)

The application is deployed and accessible for:
- **Interactive exploration** of historical price trends
- **Real-time forecasting** with custom date ranges
- **Data upload** for personalized analysis
- **Export capabilities** for further analysis

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Developed as part of Smart India Hackathon 2022 - Demonstrating practical application of statistical modeling and data visualization for real-world energy market challenges.*
