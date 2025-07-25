import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from config.settings import settings


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'  # Reset
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Add color to the level name
        record.levelname = f"{log_color}{record.levelname}{reset}"

        return super().format(record)


def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Setup logger with consistent formatting"""

    logger = logging.getLogger(name)

    # Don't add handlers if already configured
    if logger.handlers:
        return logger

    # Set log level
    log_level = level or settings.LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Create formatters
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / settings.LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "error.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)

    return logger


def setup_agent_logger(agent_id: str) -> logging.Logger:
    """Setup logger specific to an agent"""
    logger_name = f"agent.{agent_id}"
    return setup_logger(logger_name)


def setup_service_logger(service_name: str) -> logging.Logger:
    """Setup logger specific to a service"""
    logger_name = f"service.{service_name}"
    return setup_logger(logger_name)


class StructuredLogger:
    """Structured logger for JSON-style logging"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_event(self, level: str, event_type: str, message: str, **kwargs):
        """Log structured event with additional context"""
        extra_data = {
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }

        # Create structured message
        structured_msg = f"{message} | {self._format_extra(extra_data)}"

        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(structured_msg)

    def log_agent_event(self, agent_id: str, event_type: str, message: str,
                        **kwargs):
        """Log agent-specific event"""
        self.log_event(
            'info',
            f'agent_{event_type}',
            message,
            agent_id=agent_id,
            **kwargs
        )

    def log_device_event(self, device_id: str, event_type: str, message: str,
                         **kwargs):
        """Log device-specific event"""
        self.log_event(
            'info',
            f'device_{event_type}',
            message,
            device_id=device_id,
            **kwargs
        )

    def log_energy_event(self, event_type: str, message: str, **kwargs):
        """Log energy-related event"""
        self.log_event(
            'info',
            f'energy_{event_type}',
            message,
            **kwargs
        )

    def log_error(self, error: Exception, context: str = "", **kwargs):
        """Log error with context"""
        self.log_event(
            'error',
            'system_error',
            f"{context}: {str(error)}",
            error_type=type(error).__name__,
            **kwargs
        )

    def _format_extra(self, data: dict) -> str:
        """Format extra data for logging"""
        items = []
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)):
                items.append(f"{key}={value}")
            else:
                items.append(f"{key}={str(value)}")

        return " | ".join(items)


class PerformanceLogger:
    """Logger for performance monitoring"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.structured = StructuredLogger(logger)

    def log_agent_performance(self, agent_id: str, execution_time: float,
                              memory_usage: Optional[float] = None, **kwargs):
        """Log agent performance metrics"""
        self.structured.log_agent_event(
            agent_id=agent_id,
            event_type='performance',
            message=f"Agent execution completed in {execution_time:.3f}s",
            execution_time_sec=execution_time,
            memory_usage_mb=memory_usage,
            **kwargs
        )

    def log_database_performance(self, operation: str, execution_time: float,
                                 record_count: Optional[int] = None, **kwargs):
        """Log database performance metrics"""
        self.structured.log_event(
            level='debug',
            event_type='database_performance',
            message=f"Database {operation} completed in {execution_time:.3f}s",
            operation=operation,
            execution_time_sec=execution_time,
            record_count=record_count,
            **kwargs
        )

    def log_llm_performance(self, model: str, prompt_tokens: int,
                            completion_tokens: int, execution_time: float,
                            **kwargs):
        """Log LLM performance metrics"""
        self.structured.log_event(
            level='debug',
            event_type='llm_performance',
            message=f"LLM {model} request completed in {execution_time:.3f}s",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            execution_time_sec=execution_time,
            **kwargs
        )


class AuditLogger:
    """Logger for audit trail and compliance"""

    def __init__(self, logger_name: str = "audit"):
        self.logger = setup_logger(logger_name)
        self.structured = StructuredLogger(self.logger)

    def log_user_action(self, user_id: str, action: str, details: dict,
                        **kwargs):
        """Log user actions for audit trail"""
        self.structured.log_event(
            level='info',
            event_type='user_action',
            message=f"User {user_id} performed {action}",
            user_id=user_id,
            action=action,
            details=str(details),
            **kwargs
        )

    def log_system_change(self, component: str, change_type: str,
                          old_value: Any = None, new_value: Any = None,
                          **kwargs):
        """Log system configuration changes"""
        self.structured.log_event(
            level='warning',
            event_type='system_change',
            message=f"System change in {component}: {change_type}",
            component=component,
            change_type=change_type,
            old_value=str(old_value) if old_value is not None else None,
            new_value=str(new_value) if new_value is not None else None,
            **kwargs
        )

    def log_security_event(self, event_type: str, details: str,
                           severity: str = "medium", **kwargs):
        """Log security-related events"""
        self.structured.log_event(
            level='warning' if severity in ['low', 'medium'] else 'error',
            event_type='security_event',
            message=f"Security event: {event_type} - {details}",
            security_event_type=event_type,
            severity=severity,
            **kwargs
        )

    def log_data_access(self, user_id: str, data_type: str, action: str,
                        **kwargs):
        """Log data access for privacy compliance"""
        self.structured.log_event(
            level='info',
            event_type='data_access',
            message=f"Data access: {user_id} {action} {data_type}",
            user_id=user_id,
            data_type=data_type,
            action=action,
            **kwargs
        )


# Global logger instances
main_logger = setup_logger("smart_home_energy")
performance_logger = PerformanceLogger(setup_logger("performance"))
audit_logger = AuditLogger()


# Convenience functions
def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name"""
    return setup_logger(name)


def get_agent_logger(agent_id: str) -> logging.Logger:
    """Get logger for a specific agent"""
    return setup_agent_logger(agent_id)


def get_service_logger(service_name: str) -> logging.Logger:
    """Get logger for a specific service"""
    return setup_service_logger(service_name)


# Context manager for performance logging
class LogExecutionTime:
    """Context manager to log execution time"""

    def __init__(self, logger: logging.Logger, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.kwargs = kwargs
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            execution_time = (
                        datetime.utcnow() - self.start_time).total_seconds()

            if exc_type is None:
                self.logger.info(
                    f"{self.operation} completed in {execution_time:.3f}s")
            else:
                self.logger.error(
                    f"{self.operation} failed after {execution_time:.3f}s: {exc_val}")

            # Log to performance logger if available
            try:
                performance_logger.structured.log_event(
                    level='debug',
                    event_type='operation_performance',
                    message=f"{self.operation} execution time",
                    operation=self.operation,
                    execution_time_sec=execution_time,
                    success=exc_type is None,
                    **self.kwargs
                )
            except:
                pass  # Don't fail if performance logging fails