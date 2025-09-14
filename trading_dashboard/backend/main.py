"""
FastAPI Backend for Trading Strategy Dashboard
============================================
High-performance API serving real trade data with quant-level calculations
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, date

# Import custom modules
from data_processor import DataProcessor
from calculations import QuantCalculations
from models import (
    StrategyResponse,
    TradeResponse,
    FilterRequest,
    StatisticsResponse,
    EquityCurveResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Trading Strategy Dashboard API",
    description="Professional quant-level trading analysis platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data processor
data_processor = DataProcessor()
quant_calc = QuantCalculations()

# Mount static files (frontend)
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

@app.on_event("startup")
async def startup_event():
    """Load data on startup."""
    logger.info("🚀 Starting Trading Dashboard API...")
    try:
        data_processor.load_data()
        logger.info("✅ Data loaded successfully")
        logger.info(f"📊 Total trades loaded: {len(data_processor.trades_df):,}")
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise

@app.get("/")
async def root():
    """Serve the trading dashboard frontend."""
    frontend_file = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_file.exists():
        return FileResponse(frontend_file)
    else:
        # Fallback API response
        return {
            "message": "Trading Strategy Dashboard API",
            "status": "active",
            "total_trades": len(data_processor.trades_df),
            "strategies": data_processor.get_available_strategies(),
            "date_range": {
                "start": data_processor.trades_df['date'].min(),
                "end": data_processor.trades_df['date'].max()
            }
        }

@app.get("/status")
async def api_status():
    """API status endpoint."""
    return {
        "message": "Trading Strategy Dashboard API",
        "status": "active",
        "total_trades": len(data_processor.trades_df),
        "strategies": data_processor.get_available_strategies(),
        "date_range": {
            "start": data_processor.trades_df['date'].min(),
            "end": data_processor.trades_df['date'].max()
        }
    }

# Strategy Endpoints
@app.get("/api/strategies", response_model=List[StrategyResponse])
async def get_strategies():
    """Get all available strategies with summary metrics."""
    try:
        strategies = data_processor.get_strategy_summaries()
        return strategies
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/strategies/{strategy_name}/summary", response_model=StrategyResponse)
async def get_strategy_summary(strategy_name: str):
    """Get detailed summary for a specific strategy."""
    try:
        if strategy_name not in data_processor.get_available_strategies():
            raise HTTPException(status_code=404, detail="Strategy not found")

        summary = data_processor.get_strategy_summary(strategy_name)
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/strategies/{strategy_name}/trades")
async def get_strategy_trades(
    strategy_name: str,
    year: Optional[int] = Query(None, description="Filter by year"),
    weekdays: Optional[List[str]] = Query(None, description="Filter by weekdays"),
    start_date: Optional[date] = Query(None, description="Start date filter"),
    end_date: Optional[date] = Query(None, description="End date filter"),
    limit: Optional[int] = Query(1000, description="Limit number of trades"),
    offset: Optional[int] = Query(0, description="Offset for pagination")
):
    """Get trades for a specific strategy with filtering."""
    try:
        if strategy_name not in data_processor.get_available_strategies():
            raise HTTPException(status_code=404, detail="Strategy not found")

        trades = data_processor.get_filtered_trades(
            strategy_name=strategy_name,
            year=year,
            weekdays=weekdays,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )

        return {
            "strategy": strategy_name,
            "total_trades": len(trades),
            "trades": trades.to_dict('records'),
            "filters_applied": {
                "year": year,
                "weekdays": weekdays,
                "start_date": start_date,
                "end_date": end_date
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Filtering and Calculation Endpoints
@app.post("/api/filter/statistics", response_model=StatisticsResponse)
async def calculate_filtered_statistics(filter_request: FilterRequest):
    """Calculate real-time statistics based on filters."""
    try:
        filtered_df = data_processor.apply_filters(filter_request)

        if filtered_df.empty:
            raise HTTPException(status_code=404, detail="No trades match the filters")

        statistics = quant_calc.calculate_comprehensive_statistics(filtered_df)
        return statistics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating filtered statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/filter/equity-curve", response_model=EquityCurveResponse)
async def calculate_equity_curve(filter_request: FilterRequest):
    """Calculate equity curve based on filters."""
    try:
        filtered_df = data_processor.apply_filters(filter_request)

        if filtered_df.empty:
            raise HTTPException(status_code=404, detail="No trades match the filters")

        equity_curve = quant_calc.calculate_equity_curve(filtered_df)
        return equity_curve
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating equity curve: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/filter/trades")
async def get_filtered_trades(
    filter_request: FilterRequest,
    limit: int = Query(50, description="Number of trades per page"),
    offset: int = Query(0, description="Number of trades to skip")
):
    """Get filtered trade data with pagination."""
    try:
        filtered_df = data_processor.apply_filters(filter_request)

        if filtered_df.empty:
            return {"trades": [], "total_trades": 0, "page": 1, "total_pages": 0}

        # Sort by date descending
        filtered_df = filtered_df.sort_values('date', ascending=False)

        total_trades = len(filtered_df)
        total_pages = (total_trades + limit - 1) // limit
        current_page = (offset // limit) + 1

        # Apply pagination
        paginated_df = filtered_df.iloc[offset:offset + limit]

        # Convert to list of dicts
        trades = paginated_df.to_dict('records')

        return {
            "trades": trades,
            "total_trades": total_trades,
            "page": current_page,
            "total_pages": total_pages,
            "has_next": offset + limit < total_trades,
            "has_previous": offset > 0
        }
    except Exception as e:
        logger.error(f"Error getting filtered trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/filter/daily-performance")
async def calculate_daily_performance(filter_request: FilterRequest):
    """Calculate comprehensive daily performance breakdown."""
    try:
        performance_data = data_processor.get_daily_performance(filter_request)
        return performance_data
    except Exception as e:
        logger.error(f"Error calculating daily performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/filter/weekly-distribution")
async def calculate_weekly_distribution(filter_request: FilterRequest):
    """Calculate weekly points distribution analysis."""
    try:
        weekly_data = data_processor.get_weekly_distribution(filter_request)
        return weekly_data
    except Exception as e:
        logger.error(f"Error calculating weekly distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/filter/monthly-distribution")
async def calculate_monthly_distribution(filter_request: FilterRequest):
    """Calculate monthly points distribution analysis with seasonal patterns."""
    try:
        monthly_data = data_processor.get_monthly_distribution(filter_request)
        return monthly_data
    except Exception as e:
        logger.error(f"Error calculating monthly distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/calculate/expectancy")
async def calculate_expectancy(
    strategy_name: Optional[str] = Query(None, description="Strategy name"),
    year: Optional[int] = Query(None, description="Year filter"),
    weekdays: Optional[List[str]] = Query(None, description="Weekdays filter")
):
    """Calculate expectancy with optional filters."""
    try:
        filtered_df = data_processor.get_filtered_data(
            strategy_name=strategy_name,
            year=year,
            weekdays=weekdays
        )

        if filtered_df.empty:
            raise HTTPException(status_code=404, detail="No data matches the filters")

        expectancy = quant_calc.calculate_expectancy(filtered_df)
        return {"expectancy": expectancy}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating expectancy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/calculate/profit-factor")
async def calculate_profit_factor(
    strategy_name: Optional[str] = Query(None, description="Strategy name"),
    year: Optional[int] = Query(None, description="Year filter"),
    weekdays: Optional[List[str]] = Query(None, description="Weekdays filter")
):
    """Calculate profit factor with optional filters."""
    try:
        filtered_df = data_processor.get_filtered_data(
            strategy_name=strategy_name,
            year=year,
            weekdays=weekdays
        )

        if filtered_df.empty:
            raise HTTPException(status_code=404, detail="No data matches the filters")

        profit_factor = quant_calc.calculate_profit_factor(filtered_df)
        return {"profit_factor": profit_factor}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating profit factor: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/calculate/drawdown")
async def calculate_drawdown(
    strategy_name: Optional[str] = Query(None, description="Strategy name"),
    year: Optional[int] = Query(None, description="Year filter"),
    weekdays: Optional[List[str]] = Query(None, description="Weekdays filter")
):
    """Calculate maximum drawdown with optional filters."""
    try:
        filtered_df = data_processor.get_filtered_data(
            strategy_name=strategy_name,
            year=year,
            weekdays=weekdays
        )

        if filtered_df.empty:
            raise HTTPException(status_code=404, detail="No data matches the filters")

        drawdown_info = quant_calc.calculate_drawdown_analysis(filtered_df)
        return drawdown_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating drawdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Comparison Endpoints
@app.get("/api/strategies/compare")
async def compare_strategies(
    strategies: List[str] = Query(description="List of strategies to compare"),
    year: Optional[int] = Query(None, description="Year filter"),
    weekdays: Optional[List[str]] = Query(None, description="Weekdays filter")
):
    """Compare multiple strategies side by side."""
    try:
        # Validate strategies exist
        available_strategies = data_processor.get_available_strategies()
        invalid_strategies = [s for s in strategies if s not in available_strategies]
        if invalid_strategies:
            raise HTTPException(
                status_code=404,
                detail=f"Strategies not found: {invalid_strategies}"
            )

        comparison = data_processor.compare_strategies(
            strategies=strategies,
            year=year,
            weekdays=weekdays
        )

        return comparison
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check and monitoring endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": data_processor.trades_df is not None,
        "total_trades": len(data_processor.trades_df) if data_processor.trades_df is not None else 0
    }

@app.get("/api/meta/filters")
async def get_available_filters():
    """Get available filter options."""
    try:
        filters = data_processor.get_available_filters()
        return filters
    except Exception as e:
        logger.error(f"Error getting available filters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )