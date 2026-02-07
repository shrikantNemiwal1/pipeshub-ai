"""
Crawling Scheduler Service using APScheduler with Redis job store.
Replaces Node.js BullMQ-based crawling manager.
"""

import asyncio
import json
import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from app.connectors.services.kafka_service import KafkaService
from app.utils.time_conversion import get_epoch_timestamp_in_ms


class CrawlingScheduleType(str, Enum):
    """Schedule types for crawling jobs"""
    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    CUSTOM = "CUSTOM"
    ONCE = "ONCE"


class CrawlingSchedulerService:
    """
    Manages scheduled crawling jobs using APScheduler with Redis job store.
    Equivalent to Node.js BullMQ-based CrawlingSchedulerService.
    """

    def __init__(
        self,
        logger,
        kafka_service: KafkaService,
    ):
        """
        Initialize the crawling scheduler.

        Args:
            logger: Logger instance
            kafka_service: Kafka service for publishing sync events
        """
        self.logger = logger
        self.kafka_service = kafka_service

        # Use MemoryJobStore to avoid pickle serialization issues with gRPC channels
        # Note: Jobs are not persisted across restarts, but can be reloaded from database if needed
        jobstores = {
            'default': MemoryJobStore()
        }

        # Create scheduler
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            job_defaults={
                'coalesce': True,  # Combine missed runs
                'max_instances': 1,  # One instance at a time per job
                'misfire_grace_time': 300  # 5 minutes grace period
            }
        )

        # Store paused jobs (job_id -> job_info)
        self.paused_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Store job metadata for persistence (job_id -> metadata)
        self.job_metadata: Dict[str, Dict[str, Any]] = {}

        self.logger.info("CrawlingSchedulerService initialized with MemoryJobStore")

    def start(self):
        """Start the scheduler"""
        if not self.scheduler.running:
            self.scheduler.start()
            self.logger.info("Crawling scheduler started")

    def shutdown(self):
        """Shutdown the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            self.logger.info("Crawling scheduler shutdown")

    def _build_job_id(self, connector: str, connector_id: str, org_id: str) -> str:
        """
        Build consistent job ID.

        Args:
            connector: Connector name
            connector_id: Connector instance ID
            org_id: Organization ID

        Returns:
            Unique job ID string
        """
        return f"crawl-{connector.lower().replace(' ', '-')}-{connector_id}-{org_id}"

    def _transform_schedule_config(
        self,
        schedule_config: Dict[str, Any],
        timezone: str = "UTC"
    ) -> Any:
        """
        Transform schedule configuration to APScheduler trigger.

        Args:
            schedule_config: Schedule configuration dict
            timezone: Timezone string (default: UTC)

        Returns:
            APScheduler trigger object
        """
        schedule_type = schedule_config.get("scheduleType")

        if schedule_type == CrawlingScheduleType.HOURLY:
            interval = schedule_config.get("interval", 1)
            minute = schedule_config.get("minute", 0)
            return CronTrigger(
                minute=minute,
                hour=f"*/{interval}",
                timezone=timezone
            )

        elif schedule_type == CrawlingScheduleType.DAILY:
            hour = schedule_config.get("hour", 0)
            minute = schedule_config.get("minute", 0)
            return CronTrigger(
                hour=hour,
                minute=minute,
                timezone=timezone
            )

        elif schedule_type == CrawlingScheduleType.WEEKLY:
            days_of_week = schedule_config.get("daysOfWeek", [])
            hour = schedule_config.get("hour", 0)
            minute = schedule_config.get("minute", 0)
            # Convert to comma-separated string
            days_str = ",".join(str(d) for d in days_of_week)
            return CronTrigger(
                day_of_week=days_str,
                hour=hour,
                minute=minute,
                timezone=timezone
            )

        elif schedule_type == CrawlingScheduleType.MONTHLY:
            day_of_month = schedule_config.get("dayOfMonth", 1)
            hour = schedule_config.get("hour", 0)
            minute = schedule_config.get("minute", 0)
            return CronTrigger(
                day=day_of_month,
                hour=hour,
                minute=minute,
                timezone=timezone
            )

        elif schedule_type == CrawlingScheduleType.CUSTOM:
            cron_expression = schedule_config.get("cronExpression")
            if not cron_expression:
                raise ValueError("cronExpression is required for CUSTOM schedule type")
            # Parse cron expression (format: minute hour day month day_of_week)
            parts = cron_expression.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid cron expression: {cron_expression}")
            return CronTrigger.from_crontab(cron_expression, timezone=timezone)

        elif schedule_type == CrawlingScheduleType.ONCE:
            scheduled_time_str = schedule_config.get("scheduleConfig", {}).get("scheduledTime")
            if not scheduled_time_str:
                raise ValueError("scheduledTime is required for ONCE schedule type")
            # Parse ISO datetime string
            scheduled_time = datetime.fromisoformat(scheduled_time_str.replace('Z', '+00:00'))
            return DateTrigger(run_date=scheduled_time, timezone=timezone)

        else:
            raise ValueError(f"Invalid schedule type: {schedule_type}")

    async def _crawl_job_callback(
        self,
        connector: str,
        connector_id: str,
        org_id: str,
        user_id: str,
        schedule_config: Dict[str, Any]
    ):
        """
        Callback function executed when a scheduled job fires.
        Publishes sync event to Kafka for the connector.

        Args:
            connector: Connector name
            connector_id: Connector instance ID
            org_id: Organization ID
            user_id: User ID
            schedule_config: Schedule configuration
        """
        try:
            self.logger.info(
                f"Executing scheduled crawl: {connector} ({connector_id}) for org {org_id}"
            )

            # Construct sync event
            event_type = f"{connector.replace(' ', '').lower()}.resync"
            payload = {
                "orgId": org_id,
                "origin": "CONNECTOR",
                "connector": connector,
                "connectorId": connector_id,
                "createdAtTimestamp": str(get_epoch_timestamp_in_ms()),
                "updatedAtTimestamp": str(get_epoch_timestamp_in_ms()),
                "sourceCreatedAtTimestamp": str(get_epoch_timestamp_in_ms())
            }

            # Publish to sync-events Kafka topic
            message = {
                'eventType': event_type,
                'payload': payload,
                'timestamp': get_epoch_timestamp_in_ms()
            }

            await self.kafka_service.publish_event(
                topic='sync-events',
                event=message
            )

            self.logger.info(
                f"✅ Published sync event for scheduled crawl: {connector} ({connector_id})"
            )

        except Exception as e:
            self.logger.error(
                f"Error executing scheduled crawl for {connector} ({connector_id}): {e}",
                exc_info=True
            )

    async def schedule_job(
        self,
        connector: str,
        connector_id: str,
        schedule_config: Dict[str, Any],
        org_id: str,
        user_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Schedule a crawling job.

        Args:
            connector: Connector name
            connector_id: Connector instance ID
            schedule_config: Schedule configuration
            org_id: Organization ID
            user_id: User ID
            options: Optional job options (priority, maxRetries, metadata)

        Returns:
            Job information dict

        Raises:
            ValueError: If schedule is invalid or disabled
        """
        job_id = self._build_job_id(connector, connector_id, org_id)

        self.logger.info(
            f"Scheduling crawling job: {job_id} - Type: {schedule_config.get('scheduleType')}"
        )

        # Check if schedule is enabled
        if not schedule_config.get("isEnabled", False):
            self.logger.warning(f"Schedule is disabled, not creating job: {job_id}")
            raise ValueError("Cannot schedule a disabled job")

        # Remove any existing job
        await self.remove_job(connector, connector_id, org_id)

        # Remove from paused jobs if it exists
        if job_id in self.paused_jobs:
            del self.paused_jobs[job_id]

        # Get timezone
        timezone = schedule_config.get("timezone", "UTC")

        # Transform schedule configuration to trigger
        trigger = self._transform_schedule_config(schedule_config, timezone)

        # Store job metadata for potential persistence
        self.job_metadata[job_id] = {
            "connector": connector,
            "connectorId": connector_id,
            "orgId": org_id,
            "userId": user_id,
            "scheduleConfig": schedule_config,
            "options": options or {},
            "createdAt": datetime.now().isoformat()
        }

        # Add job to scheduler
        job = self.scheduler.add_job(
            func=self._crawl_job_callback,
            trigger=trigger,
            args=[connector, connector_id, org_id, user_id, schedule_config],
            id=job_id,
            name=f"crawl-{connector.lower()}-{connector_id}",
            replace_existing=True,
            kwargs={}
        )

        self.logger.info(f"✅ Crawling job scheduled successfully: {job_id}")

        # Get next run time from the job
        next_run_time = None
        try:
            # In APScheduler, next_run_time might be None initially or accessed differently
            if hasattr(job, 'next_run_time') and job.next_run_time:
                next_run_time = job.next_run_time.isoformat()
        except (AttributeError, TypeError) as e:
            self.logger.debug(f"Could not get next_run_time from job: {e}")
            # Try to get it from the scheduler
            try:
                scheduled_job = self.scheduler.get_job(job_id)
                if scheduled_job and scheduled_job.next_run_time:
                    next_run_time = scheduled_job.next_run_time.isoformat()
            except Exception as e2:
                self.logger.debug(f"Could not get next_run_time from scheduler: {e2}")

        return {
            "jobId": job_id,
            "connector": connector,
            "connectorId": connector_id,
            "scheduleConfig": schedule_config,
            "scheduledAt": datetime.now().isoformat(),
            "nextRunTime": next_run_time
        }

    async def remove_job(
        self,
        connector: str,
        connector_id: str,
        org_id: str
    ):
        """
        Remove a scheduled job.

        Args:
            connector: Connector name
            connector_id: Connector instance ID
            org_id: Organization ID
        """
        job_id = self._build_job_id(connector, connector_id, org_id)

        self.logger.info(f"Removing crawling job: {job_id}")

        try:
            # Remove from scheduler
            if self.scheduler.get_job(job_id):
                self.scheduler.remove_job(job_id)
                self.logger.info(f"✅ Removed job from scheduler: {job_id}")

            # Remove from paused jobs
            if job_id in self.paused_jobs:
                del self.paused_jobs[job_id]
                self.logger.info(f"✅ Removed job from paused jobs: {job_id}")
            
            # Remove from job metadata
            if job_id in self.job_metadata:
                del self.job_metadata[job_id]
                self.logger.info(f"✅ Removed job metadata: {job_id}")

        except Exception as e:
            self.logger.warning(f"Error removing job {job_id}: {e}")

    async def get_job_status(
        self,
        connector: str,
        connector_id: str,
        org_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get status of a scheduled job.

        Args:
            connector: Connector name
            connector_id: Connector instance ID
            org_id: Organization ID

        Returns:
            Job status dict or None if not found
        """
        job_id = self._build_job_id(connector, connector_id, org_id)

        self.logger.debug(f"Getting job status: {job_id}")

        # Check if job is paused
        if job_id in self.paused_jobs:
            paused_info = self.paused_jobs[job_id]
            return {
                "id": job_id,
                "name": f"crawl-{connector.lower()}-{connector_id}",
                "data": {
                    "connector": connector,
                    "connectorId": connector_id,
                    "scheduleConfig": paused_info["scheduleConfig"],
                    "orgId": org_id,
                    "userId": paused_info["userId"]
                },
                "state": "paused",
                "timestamp": paused_info["pausedAt"],
                "nextRunTime": None
            }

        # Get job from scheduler
        job = self.scheduler.get_job(job_id)

        if not job:
            self.logger.debug(f"No job found: {job_id}")
            return None

        # Safely get next_run_time
        next_run_time = None
        if hasattr(job, 'next_run_time') and job.next_run_time:
            try:
                next_run_time = job.next_run_time.isoformat()
            except (AttributeError, TypeError):
                pass

        return {
            "id": job_id,
            "name": job.name,
            "state": "scheduled",
            "nextRunTime": next_run_time,
            "data": {
                "connector": connector,
                "connectorId": connector_id,
                "orgId": org_id
            }
        }

    async def get_all_jobs(self, org_id: str) -> List[Dict[str, Any]]:
        """
        Get all jobs for an organization.

        Args:
            org_id: Organization ID

        Returns:
            List of job status dicts
        """
        self.logger.debug(f"Getting all jobs for organization: {org_id}")

        job_statuses = []

        # Get all scheduled jobs
        all_jobs = self.scheduler.get_jobs()

        for job in all_jobs:
            # Check if job belongs to this org (job_id format: crawl-{connector}-{connectorId}-{orgId})
            if job.id.endswith(f"-{org_id}"):
                # Safely get next_run_time
                next_run_time = None
                if hasattr(job, 'next_run_time') and job.next_run_time:
                    try:
                        next_run_time = job.next_run_time.isoformat()
                    except (AttributeError, TypeError):
                        pass
                
                job_statuses.append({
                    "id": job.id,
                    "name": job.name,
                    "state": "scheduled",
                    "nextRunTime": next_run_time
                })

        # Add paused jobs for this org
        for job_id, paused_info in self.paused_jobs.items():
            if paused_info.get("orgId") == org_id:
                job_statuses.append({
                    "id": job_id,
                    "name": f"crawl-{paused_info['connector'].lower()}-{paused_info['connectorId']}",
                    "state": "paused",
                    "timestamp": paused_info["pausedAt"],
                    "data": {
                        "connector": paused_info["connector"],
                        "connectorId": paused_info["connectorId"],
                        "orgId": paused_info["orgId"],
                        "userId": paused_info["userId"]
                    }
                })

        self.logger.debug(f"Found {len(job_statuses)} jobs for org {org_id}")

        return job_statuses

    async def remove_all_jobs(self, org_id: str):
        """
        Remove all jobs for an organization.

        Args:
            org_id: Organization ID
        """
        self.logger.info(f"Removing all jobs for organization: {org_id}")

        # Get all jobs for this org
        all_jobs = self.scheduler.get_jobs()

        removed_count = 0
        for job in all_jobs:
            # Check if job belongs to this org
            if job.id.endswith(f"-{org_id}"):
                try:
                    self.scheduler.remove_job(job.id)
                    # Remove from job metadata
                    if job.id in self.job_metadata:
                        del self.job_metadata[job.id]
                    removed_count += 1
                except Exception as e:
                    self.logger.warning(f"Error removing job {job.id}: {e}")

        # Remove paused jobs for this org
        paused_to_remove = [
            job_id for job_id, info in self.paused_jobs.items()
            if info.get("orgId") == org_id
        ]

        for job_id in paused_to_remove:
            del self.paused_jobs[job_id]
            removed_count += 1

        self.logger.info(f"✅ Removed {removed_count} jobs for org {org_id}")

    async def pause_job(
        self,
        connector: str,
        connector_id: str,
        org_id: str
    ):
        """
        Pause a scheduled job.

        Args:
            connector: Connector name
            connector_id: Connector instance ID
            org_id: Organization ID

        Raises:
            ValueError: If job not found or already paused
        """
        job_id = self._build_job_id(connector, connector_id, org_id)

        self.logger.info(f"Pausing job: {job_id}")

        # Check if already paused
        if job_id in self.paused_jobs:
            raise ValueError("Job is already paused")

        # Get current job
        job = self.scheduler.get_job(job_id)

        if not job:
            raise ValueError("No active job found to pause")

        # Store job info
        # Extract schedule config from job args
        job_args = job.args if hasattr(job, 'args') else []
        schedule_config = job_args[4] if len(job_args) > 4 else {}
        user_id = job_args[3] if len(job_args) > 3 else "unknown"

        self.paused_jobs[job_id] = {
            "connector": connector,
            "connectorId": connector_id,
            "orgId": org_id,
            "userId": user_id,
            "scheduleConfig": schedule_config,
            "pausedAt": datetime.now().isoformat()
        }

        # Remove from scheduler
        self.scheduler.remove_job(job_id)

        self.logger.info(f"✅ Job paused successfully: {job_id}")

    async def resume_job(
        self,
        connector: str,
        connector_id: str,
        org_id: str
    ):
        """
        Resume a paused job.

        Args:
            connector: Connector name
            connector_id: Connector instance ID
            org_id: Organization ID

        Raises:
            ValueError: If job not found in paused jobs
        """
        job_id = self._build_job_id(connector, connector_id, org_id)

        self.logger.info(f"Resuming job: {job_id}")

        # Get paused job info
        if job_id not in self.paused_jobs:
            raise ValueError("No paused job found to resume")

        paused_info = self.paused_jobs[job_id]

        # Re-schedule the job
        await self.schedule_job(
            connector=paused_info["connector"],
            connector_id=paused_info["connectorId"],
            schedule_config=paused_info["scheduleConfig"],
            org_id=paused_info["orgId"],
            user_id=paused_info["userId"]
        )

        # Remove from paused jobs
        del self.paused_jobs[job_id]

        self.logger.info(f"✅ Job resumed successfully: {job_id}")

    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.

        Returns:
            Statistics dict
        """
        all_jobs = self.scheduler.get_jobs()

        return {
            "scheduled": len(all_jobs),
            "paused": len(self.paused_jobs),
            "total": len(all_jobs) + len(self.paused_jobs),
            "running": self.scheduler.running,
            "metadata_stored": len(self.job_metadata)
        }
    
    def get_all_job_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all stored job metadata.
        
        Returns:
            Dictionary of job_id -> metadata
        """
        return self.job_metadata.copy()

