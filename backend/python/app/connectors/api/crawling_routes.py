"""
Crawling Manager REST API endpoints.
Replaces Node.js BullMQ-based crawling manager.
"""

from typing import Any, Dict, Optional

from app.config.constants.http_status_code import HttpStatusCode
from app.connectors.services.crawling_scheduler import (
    CrawlingSchedulerService,
    CrawlingScheduleType,
)
from app.containers.connector import ConnectorAppContainer
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class ScheduleConfig(BaseModel):
    """Schedule configuration model"""
    scheduleType: CrawlingScheduleType
    isEnabled: bool = True
    timezone: str = "UTC"
    
    # Hourly schedule fields
    interval: Optional[int] = None  # For HOURLY
    minute: Optional[int] = None  # For HOURLY/DAILY/WEEKLY/MONTHLY
    
    # Daily/Weekly/Monthly schedule fields
    hour: Optional[int] = None  # For DAILY/WEEKLY/MONTHLY
    
    # Weekly schedule fields
    daysOfWeek: Optional[list[int]] = None  # For WEEKLY (0-6)
    
    # Monthly schedule fields
    dayOfMonth: Optional[int] = None  # For MONTHLY (1-31)
    
    # Custom schedule fields
    cronExpression: Optional[str] = None  # For CUSTOM
    
    # Once schedule fields
    scheduleConfig: Optional[Dict[str, Any]] = None  # For ONCE (contains scheduledTime)
    
    @model_validator(mode='before')
    @classmethod
    def normalize_schedule_type(cls, data: Any) -> Any:
        """Normalize scheduleType to uppercase for case-insensitive validation"""
        if isinstance(data, dict) and 'scheduleType' in data:
            data['scheduleType'] = data['scheduleType'].upper()
        return data


class ScheduleCrawlingRequest(BaseModel):
    """Request body for scheduling a crawling job"""
    scheduleConfig: ScheduleConfig
    priority: Optional[int] = Field(default=5, ge=1, le=10)
    maxRetries: Optional[int] = Field(default=3, ge=0, le=10)
    metadata: Optional[Dict[str, Any]] = None


class ValidationSchema(BaseModel):
    """Validation schema for connector parameters"""
    connector: str
    connectorId: str


# ============================================================================
# Helper Functions
# ============================================================================

def _get_user_context(request: Request) -> Dict[str, str]:
    """Extract user context from request"""
    user_id = request.state.user.get("userId", "unknown")
    org_id = request.state.user.get("orgId", "unknown")
    is_admin = request.state.user.get("isAdmin", False)
    
    return {
        "userId": user_id,
        "orgId": org_id,
        "isAdmin": is_admin
    }


async def _validate_connector_access(
    request: Request,
    connector_id: str,
    config_service,
    logger
):
    """
    Validate that the user has access to the connector.
    Team scope: admin only
    Personal scope: creator only
    
    Note: For now, we'll allow access since connector validation
    would require calling the connectors API. This can be enhanced later.
    """
    # TODO: Call connector API to validate access if needed
    pass


# ============================================================================
# Crawling Manager Endpoints
# ============================================================================

@router.post("/api/v1/connectors/crawling/{connector}/{connectorId}/schedule")
@inject
async def schedule_crawling_job(
    connector: str,
    connectorId: str,
    request: Request,
    body: ScheduleCrawlingRequest,
    scheduler: CrawlingSchedulerService = Depends(Provide[ConnectorAppContainer.crawling_scheduler])
) -> Dict[str, Any]:
    """
    Schedule a crawling job for a specific connector.
    
    Args:
        connector: Connector type name
        connectorId: Connector instance ID
        request: FastAPI request object
        body: Schedule configuration request body
        scheduler: Injected crawling scheduler service
    
    Returns:
        Dictionary with job details
    
    Raises:
        HTTPException: 400 if schedule is invalid, 403 if unauthorized
    """
    container = request.app.container
    logger = container.logger()
    
    try:
        user_context = _get_user_context(request)
        
        logger.info(
            f"Scheduling crawling job: {connector} ({connectorId}) "
            f"for org {user_context['orgId']}, scheduleType: {body.scheduleConfig.scheduleType}"
        )
        
        # Validate connector access
        config_service = container.config_service()
        await _validate_connector_access(request, connectorId, config_service, logger)
        
        # Schedule the job
        schedule_config_dict = body.scheduleConfig.dict()
        
        job_info = await scheduler.schedule_job(
            connector=connector,
            connector_id=connectorId,
            schedule_config=schedule_config_dict,
            org_id=user_context["orgId"],
            user_id=user_context["userId"],
            options={
                "priority": body.priority,
                "maxRetries": body.maxRetries,
                "metadata": body.metadata
            }
        )
        
        logger.info(f"✅ Crawling job scheduled: {job_info['jobId']}")
        
        return {
            "success": True,
            "message": "Crawling job scheduled successfully",
            "data": job_info
        }
        
    except ValueError as e:
        logger.error(f"Invalid schedule configuration: {e}")
        raise HTTPException(
            status_code=HttpStatusCode.BAD_REQUEST.value,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling crawling job: {e}", exc_info=True)
        raise HTTPException(
            status_code=HttpStatusCode.INTERNAL_SERVER_ERROR.value,
            detail=f"Failed to schedule crawling job: {str(e)}"
        )


@router.get("/api/v1/connectors/crawling/{connector}/{connectorId}/schedule")
@inject
async def get_crawling_job_status(
    connector: str,
    connectorId: str,
    request: Request,
    scheduler: CrawlingSchedulerService = Depends(Provide[ConnectorAppContainer.crawling_scheduler])
) -> Dict[str, Any]:
    """
    Get status of a scheduled crawling job.
    
    Args:
        connector: Connector type name
        connectorId: Connector instance ID
        request: FastAPI request object
        scheduler: Injected crawling scheduler service
    
    Returns:
        Dictionary with job status
    
    Raises:
        HTTPException: 404 if job not found
    """
    container = request.app.container
    logger = container.logger()
    
    try:
        user_context = _get_user_context(request)
        
        # Validate connector access
        config_service = container.config_service()
        await _validate_connector_access(request, connectorId, config_service, logger)
        
        # Get job status
        job_status = await scheduler.get_job_status(
            connector=connector,
            connector_id=connectorId,
            org_id=user_context["orgId"]
        )
        
        if not job_status:
            return {
                "success": False,
                "message": "No scheduled job found for this connector",
                "data": None
            }
        
        return {
            "success": True,
            "message": "Job status retrieved successfully",
            "data": job_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}", exc_info=True)
        raise HTTPException(
            status_code=HttpStatusCode.INTERNAL_SERVER_ERROR.value,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.get("/api/v1/connectors/crawling/schedule/all")
@inject
async def get_all_crawling_jobs(
    request: Request,
    scheduler: CrawlingSchedulerService = Depends(Provide[ConnectorAppContainer.crawling_scheduler])
) -> Dict[str, Any]:
    """
    Get all scheduled crawling jobs for the organization.
    
    Args:
        request: FastAPI request object
        scheduler: Injected crawling scheduler service
    
    Returns:
        Dictionary with list of job statuses
    """
    container = request.app.container
    logger = container.logger()
    
    try:
        user_context = _get_user_context(request)
        
        # Get all jobs for this org
        job_statuses = await scheduler.get_all_jobs(org_id=user_context["orgId"])
        
        return {
            "success": True,
            "message": "All job statuses retrieved successfully",
            "data": job_statuses
        }
        
    except Exception as e:
        logger.error(f"Error getting all job statuses: {e}", exc_info=True)
        raise HTTPException(
            status_code=HttpStatusCode.INTERNAL_SERVER_ERROR.value,
            detail=f"Failed to get all job statuses: {str(e)}"
        )


@router.delete("/api/v1/connectors/crawling/schedule/all")
@inject
async def remove_all_crawling_jobs(
    request: Request,
    scheduler: CrawlingSchedulerService = Depends(Provide[ConnectorAppContainer.crawling_scheduler])
) -> Dict[str, Any]:
    """
    Remove all scheduled crawling jobs for the organization.
    
    Args:
        request: FastAPI request object
        scheduler: Injected crawling scheduler service
    
    Returns:
        Dictionary with success status
    """
    container = request.app.container
    logger = container.logger()
    
    try:
        user_context = _get_user_context(request)
        
        logger.info(f"Removing all crawling jobs for org {user_context['orgId']}")
        
        # Remove all jobs for this org
        await scheduler.remove_all_jobs(org_id=user_context["orgId"])
        
        logger.info(f"✅ All crawling jobs removed for org {user_context['orgId']}")
        
        return {
            "success": True,
            "message": "All crawling jobs removed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error removing all jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=HttpStatusCode.INTERNAL_SERVER_ERROR.value,
            detail=f"Failed to remove all jobs: {str(e)}"
        )


@router.delete("/api/v1/connectors/crawling/{connector}/{connectorId}/remove")
@inject
async def remove_crawling_job(
    connector: str,
    connectorId: str,
    request: Request,
    scheduler: CrawlingSchedulerService = Depends(Provide[ConnectorAppContainer.crawling_scheduler])
) -> Dict[str, Any]:
    """
    Remove a specific scheduled crawling job.
    
    Args:
        connector: Connector type name
        connectorId: Connector instance ID
        request: FastAPI request object
        scheduler: Injected crawling scheduler service
    
    Returns:
        Dictionary with success status
    """
    container = request.app.container
    logger = container.logger()
    
    try:
        user_context = _get_user_context(request)
        
        # Validate connector access
        config_service = container.config_service()
        await _validate_connector_access(request, connectorId, config_service, logger)
        
        logger.info(f"Removing crawling job: {connector} ({connectorId})")
        
        # Remove the job
        await scheduler.remove_job(
            connector=connector,
            connector_id=connectorId,
            org_id=user_context["orgId"]
        )
        
        logger.info(f"✅ Crawling job removed: {connector} ({connectorId})")
        
        return {
            "success": True,
            "message": "Crawling job removed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing crawling job: {e}", exc_info=True)
        raise HTTPException(
            status_code=HttpStatusCode.INTERNAL_SERVER_ERROR.value,
            detail=f"Failed to remove crawling job: {str(e)}"
        )


@router.post("/api/v1/connectors/crawling/{connector}/{connectorId}/pause")
@inject
async def pause_crawling_job(
    connector: str,
    connectorId: str,
    request: Request,
    scheduler: CrawlingSchedulerService = Depends(Provide[ConnectorAppContainer.crawling_scheduler])
) -> Dict[str, Any]:
    """
    Pause a scheduled crawling job.
    
    Args:
        connector: Connector type name
        connectorId: Connector instance ID
        request: FastAPI request object
        scheduler: Injected crawling scheduler service
    
    Returns:
        Dictionary with success status
    
    Raises:
        HTTPException: 400 if job not found or already paused
    """
    container = request.app.container
    logger = container.logger()
    
    try:
        user_context = _get_user_context(request)
        
        # Validate connector access
        config_service = container.config_service()
        await _validate_connector_access(request, connectorId, config_service, logger)
        
        logger.info(f"Pausing crawling job: {connector} ({connectorId})")
        
        # Pause the job
        await scheduler.pause_job(
            connector=connector,
            connector_id=connectorId,
            org_id=user_context["orgId"]
        )
        
        logger.info(f"✅ Crawling job paused: {connector} ({connectorId})")
        
        return {
            "success": True,
            "message": "Crawling job paused successfully",
            "data": {
                "connector": connector,
                "connectorId": connectorId,
                "orgId": user_context["orgId"],
                "pausedAt": "now"  # Can enhance with actual timestamp
            }
        }
        
    except ValueError as e:
        logger.error(f"Error pausing job: {e}")
        raise HTTPException(
            status_code=HttpStatusCode.BAD_REQUEST.value,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing crawling job: {e}", exc_info=True)
        raise HTTPException(
            status_code=HttpStatusCode.INTERNAL_SERVER_ERROR.value,
            detail=f"Failed to pause crawling job: {str(e)}"
        )


@router.post("/api/v1/connectors/crawling/{connector}/{connectorId}/resume")
@inject
async def resume_crawling_job(
    connector: str,
    connectorId: str,
    request: Request,
    scheduler: CrawlingSchedulerService = Depends(Provide[ConnectorAppContainer.crawling_scheduler])
) -> Dict[str, Any]:
    """
    Resume a paused crawling job.
    
    Args:
        connector: Connector type name
        connectorId: Connector instance ID
        request: FastAPI request object
        scheduler: Injected crawling scheduler service
    
    Returns:
        Dictionary with success status
    
    Raises:
        HTTPException: 400 if job not found in paused jobs
    """
    container = request.app.container
    logger = container.logger()
    
    try:
        user_context = _get_user_context(request)
        
        # Validate connector access
        config_service = container.config_service()
        await _validate_connector_access(request, connectorId, config_service, logger)
        
        logger.info(f"Resuming crawling job: {connector} ({connectorId})")
        
        # Resume the job
        await scheduler.resume_job(
            connector=connector,
            connector_id=connectorId,
            org_id=user_context["orgId"]
        )
        
        logger.info(f"✅ Crawling job resumed: {connector} ({connectorId})")
        
        return {
            "success": True,
            "message": "Crawling job resumed successfully",
            "data": {
                "connector": connector,
                "connectorId": connectorId,
                "orgId": user_context["orgId"],
                "resumedAt": "now"  # Can enhance with actual timestamp
            }
        }
        
    except ValueError as e:
        logger.error(f"Error resuming job: {e}")
        raise HTTPException(
            status_code=HttpStatusCode.BAD_REQUEST.value,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming crawling job: {e}", exc_info=True)
        raise HTTPException(
            status_code=HttpStatusCode.INTERNAL_SERVER_ERROR.value,
            detail=f"Failed to resume crawling job: {str(e)}"
        )


@router.get("/api/v1/connectors/crawling/stats")
@inject
async def get_queue_stats(
    request: Request,
    scheduler: CrawlingSchedulerService = Depends(Provide[ConnectorAppContainer.crawling_scheduler])
) -> Dict[str, Any]:
    """
    Get scheduler statistics.
    
    Args:
        request: FastAPI request object
        scheduler: Injected crawling scheduler service
    
    Returns:
        Dictionary with queue statistics
    """
    container = request.app.container
    logger = container.logger()
    
    try:
        # Get queue stats
        stats = await scheduler.get_queue_stats()
        
        return {
            "success": True,
            "message": "Queue statistics retrieved successfully",
            "data": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=HttpStatusCode.INTERNAL_SERVER_ERROR.value,
            detail=f"Failed to get queue statistics: {str(e)}"
        )

