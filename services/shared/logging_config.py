"""Structured logging configuration for the Moondream Vision Pipeline."""

import logging
import logging.config
import sys
from typing import Any, Dict
import json
from datetime import datetime

from .config import get_config


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add component-specific fields
        if hasattr(record, "component"):
            log_data["component"] = record.component
        
        if hasattr(record, "frame_id"):
            log_data["frame_id"] = record.frame_id
            
        if hasattr(record, "processing_time"):
            log_data["processing_time_ms"] = record.processing_time
            
        if hasattr(record, "model_name"):
            log_data["model_name"] = record.model_name
        
        return json.dumps(log_data, ensure_ascii=False)


class ComponentLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds component context to log records."""
    
    def __init__(self, logger: logging.Logger, component: str):
        self.component = component
        super().__init__(logger, {"component": component})
    
    def process(self, msg: Any, kwargs: Dict[str, Any]) -> tuple:
        """Process log record to add component context."""
        extra = kwargs.get("extra", {})
        extra["component"] = self.component
        kwargs["extra"] = extra
        return msg, kwargs


def setup_logging(component: str = "moondream-vision") -> ComponentLoggerAdapter:
    """Set up structured logging for a component.
    
    Args:
        component: Name of the component (e.g., "camera", "yolo", "moondream")
        
    Returns:
        ComponentLoggerAdapter: Logger instance with component context
    """
    config = get_config()
    
    # Configure root logger
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JSONFormatter,
            },
            "text": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": config.log_format,
                "level": config.log_level,
            },
        },
        "root": {
            "level": config.log_level,
            "handlers": ["console"],
        },
        "loggers": {
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "fastapi": {
                "level": "INFO", 
                "handlers": ["console"],
                "propagate": False,
            },
        },
    }
    
    logging.config.dictConfig(logging_config)
    
    # Create component logger
    logger = logging.getLogger(f"moondream.{component}")
    return ComponentLoggerAdapter(logger, component)


def log_performance_metrics(
    logger: ComponentLoggerAdapter,
    operation: str,
    processing_time: float,
    frame_id: int = None,
    model_name: str = None,
    **kwargs
):
    """Log performance metrics in a structured format.
    
    Args:
        logger: Logger instance
        operation: Name of the operation (e.g., "yolo_inference", "frame_capture")
        processing_time: Processing time in milliseconds
        frame_id: Optional frame ID
        model_name: Optional model name
        **kwargs: Additional metrics to log
    """
    extra = {
        "operation": operation,
        "processing_time": processing_time,
        **kwargs
    }
    
    if frame_id is not None:
        extra["frame_id"] = frame_id
        
    if model_name is not None:
        extra["model_name"] = model_name
    
    logger.info(
        f"Performance: {operation} completed in {processing_time:.2f}ms",
        extra=extra
    )


def log_error_with_context(
    logger: ComponentLoggerAdapter,
    error: Exception,
    context: Dict[str, Any] = None,
    operation: str = None
):
    """Log an error with additional context.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information
        operation: Operation that failed
    """
    extra = {
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    
    if context:
        extra["context"] = context
        
    if operation:
        extra["operation"] = operation
    
    logger.error(
        f"Error in {operation or 'operation'}: {error}",
        extra=extra,
        exc_info=True
    )
