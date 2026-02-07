import { Router, Response, NextFunction } from 'express';
import { Container } from 'inversify';
import { metricsMiddleware } from '../../../libs/middlewares/prometheus.middleware';
import { AuthMiddleware } from '../../../libs/middlewares/auth.middleware';
import { AppConfig } from '../../tokens_manager/config/config';
import axios from 'axios';
import { Logger } from '../../../libs/services/logger.service';
import { AuthenticatedUserRequest } from '../../../libs/middlewares/types';

const logger = Logger.getInstance({ service: 'CrawlingManagerProxy' });

/**
 * Transform frontend scheduling request to Python APScheduler format
 */
function transformScheduleRequest(body: any): any {
  if (!body || !body.sync) {
    return body; // Pass through if not a schedule request
  }

  const { sync, filters, baseUrl, ...rest } = body;

  // Handle "intervalMinutes" format from frontend
  if (sync.scheduledConfig?.intervalMinutes !== undefined) {
    const minutes = sync.scheduledConfig.intervalMinutes;
    
    // For intervals < 60 minutes, use CUSTOM with cron expression
    if (minutes < 60) {
      return {
        scheduleConfig: {
          scheduleType: 'CUSTOM',
          cronExpression: `*/${minutes} * * * *`,
          isEnabled: sync.selectedStrategy === 'SCHEDULED',
          timezone: sync.scheduledConfig.timezone || 'UTC',
        },
        priority: 5,
        maxRetries: 3,
        metadata: {
          filters,
          baseUrl,
          originalFormat: 'frontend-v1',
        },
        ...rest,
      };
    } else {
      // For 60+ minutes, use HOURLY schedule
      const hours = Math.floor(minutes / 60);
      return {
        scheduleConfig: {
          scheduleType: 'HOURLY',
          interval: hours,
          minute: 0,
          isEnabled: sync.selectedStrategy === 'SCHEDULED',
          timezone: sync.scheduledConfig.timezone || 'UTC',
        },
        priority: 5,
        maxRetries: 3,
        metadata: {
          filters,
          baseUrl,
          originalFormat: 'frontend-v1',
        },
        ...rest,
      };
    }
  }

  // If already in correct format, return as-is
  if (body.scheduleConfig) {
    return body;
  }

  logger.warn('Unknown schedule request format, passing through', { body });
  return body;
}

/**
 * Proxy middleware that forwards crawling manager requests to Python connector service.
 * All crawling scheduling is now handled by Python APScheduler.
 */
function createProxyMiddleware(appConfig: AppConfig) {
  return async (req: AuthenticatedUserRequest, res: Response, _next: NextFunction) => {
    try {
      const pythonUrl = `${appConfig.connectorBackend}/api/v1/connectors/crawling${req.path}`;
      
      logger.info(`Proxying crawling manager request to Python: ${pythonUrl}`, {
        method: req.method,
        path: req.path,
      });

      // Transform request body if it's a schedule request
      let requestBody = req.body;
      if (req.method === 'POST' && req.path.includes('/schedule')) {
        requestBody = transformScheduleRequest(req.body);
        logger.info('Transformed schedule request body', { 
          original: req.body,
          transformed: requestBody 
        });
      }

      // Forward the request to Python
      const response = await axios({
        method: req.method as any,
        url: pythonUrl,
        data: requestBody,
        params: req.query,
        headers: {
          'Content-Type': 'application/json',
          // Forward Authorization header for Python auth middleware
          'Authorization': req.headers.authorization || '',
          // Forward user context for Python
          'X-User-Id': req.user?.userId || '',
          'X-Org-Id': req.user?.orgId || '',
          'X-Is-Admin': req.user?.isAdmin ? 'true' : 'false',
        },
        timeout: 30000, // 30 second timeout
      });

      // Forward the Python response back to the client
      res.status(response.status).json(response.data);
    } catch (error: any) {
      logger.error('Error proxying crawling manager request to Python', {
        error: error.message,
        status: error.response?.status,
        data: error.response?.data,
      });

      // Forward error response from Python if available
      if (error.response) {
        res.status(error.response.status).json(error.response.data);
      } else {
        res.status(500).json({
          success: false,
          message: 'Failed to communicate with crawling scheduler service',
          error: error.message,
        });
      }
    }
  };
}

export function createCrawlingManagerRouter(container: Container): Router {
  const router = Router();
  const appConfig = container.get<AppConfig>('AppConfig');
  const authMiddleware = container.get<AuthMiddleware>(AuthMiddleware);
  const proxyMiddleware = createProxyMiddleware(appConfig);

  // All routes now proxy to Python connector service
  
  // POST /api/v1/crawlingManager/:connector/:connectorId/schedule
  router.post(
    '/:connector/:connectorId/schedule',
    authMiddleware.authenticate,
    metricsMiddleware(container),
    proxyMiddleware,
  );

  // GET /api/v1/crawlingManager/:connector/:connectorId/schedule
  router.get(
    '/:connector/:connectorId/schedule',
    authMiddleware.authenticate,
    metricsMiddleware(container),
    proxyMiddleware,
  );

  // GET /api/v1/crawlingManager/schedule/all
  router.get(
    '/schedule/all',
    authMiddleware.authenticate,
    metricsMiddleware(container),
    proxyMiddleware,
  );

  // DELETE /api/v1/crawlingManager/schedule/all
  router.delete(
    '/schedule/all',
    authMiddleware.authenticate,
    metricsMiddleware(container),
    proxyMiddleware,
  );

  // DELETE /api/v1/crawlingManager/:connector/:connectorId/remove
  router.delete(
    '/:connector/:connectorId/remove',
    authMiddleware.authenticate,
    metricsMiddleware(container),
    proxyMiddleware,
  );

  // POST /api/v1/crawlingManager/:connector/:connectorId/pause
  router.post(
    '/:connector/:connectorId/pause',
    authMiddleware.authenticate,
    metricsMiddleware(container),
    proxyMiddleware,
  );

  // POST /api/v1/crawlingManager/:connector/:connectorId/resume
  router.post(
    '/:connector/:connectorId/resume',
    authMiddleware.authenticate,
    metricsMiddleware(container),
    proxyMiddleware,
  );

  // GET /api/v1/crawlingManager/stats
  router.get(
    '/stats',
    authMiddleware.authenticate,
    metricsMiddleware(container),
    proxyMiddleware,
  );

  return router;
}

export default createCrawlingManagerRouter;