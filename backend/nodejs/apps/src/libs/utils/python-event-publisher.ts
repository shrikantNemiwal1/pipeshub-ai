/**
 * Utility for publishing events to Python connector service.
 * Replaces Kafka publishing from Node.js.
 */

import axios, { AxiosError } from 'axios';
import { Logger } from '../services/logger.service';

const logger = Logger.getInstance({ service: 'PythonEventPublisher' });

export interface EventPayload {
  eventType: string;
  payload: Record<string, any>;
}

/**
 * Publish an entity event to Python connector service.
 * Replaces direct Kafka publishing for org/user/app events.
 * 
 * @param connectorBackend - Python connector service base URL
 * @param eventType - Event type (e.g., 'orgCreated', 'userAdded', 'appEnabled')
 * @param payload - Event payload
 */
export async function publishEntityEvent(
  connectorBackend: string,
  eventType: string,
  payload: Record<string, any>
): Promise<void> {
  try {
    const url = `${connectorBackend}/api/v1/connectors/entity-event`;
    const data: EventPayload = { eventType, payload };

    logger.info(`Publishing entity event to Python: ${eventType}`, {
      url,
      eventType,
    });

    const response = await axios.post(url, data, {
      headers: { 'Content-Type': 'application/json' },
      timeout: 10000, // 10 second timeout
    });

    if (response.status === 200 && response.data.success) {
      logger.info(`✅ Entity event published successfully: ${eventType}`);
    } else {
      logger.error(`Failed to publish entity event: ${eventType}`, {
        status: response.status,
        data: response.data,
      });
      throw new Error(`Failed to publish entity event: ${eventType}`);
    }
  } catch (error) {
    if (error instanceof AxiosError) {
      logger.error(`Error publishing entity event: ${eventType}`, {
        message: error.message,
        status: error.response?.status,
        data: error.response?.data,
      });
    } else {
      logger.error(`Error publishing entity event: ${eventType}`, { error });
    }
    throw error;
  }
}

/**
 * Publish a sync event to Python connector service.
 * Replaces direct Kafka publishing for Gmail/connector sync events.
 * 
 * @param connectorBackend - Python connector service base URL
 * @param eventType - Event type (e.g., 'gmailUpdatesEnabledEvent', 'connectorPublicUrlChanged')
 * @param payload - Event payload
 */
export async function publishSyncEvent(
  connectorBackend: string,
  eventType: string,
  payload: Record<string, any>
): Promise<void> {
  try {
    const url = `${connectorBackend}/api/v1/connectors/sync-event`;
    const data: EventPayload = { eventType, payload };

    logger.info(`Publishing sync event to Python: ${eventType}`, {
      url,
      eventType,
    });

    const response = await axios.post(url, data, {
      headers: { 'Content-Type': 'application/json' },
      timeout: 10000,
    });

    if (response.status === 200 && response.data.success) {
      logger.info(`✅ Sync event published successfully: ${eventType}`);
    } else {
      logger.error(`Failed to publish sync event: ${eventType}`, {
        status: response.status,
        data: response.data,
      });
      throw new Error(`Failed to publish sync event: ${eventType}`);
    }
  } catch (error) {
    if (error instanceof AxiosError) {
      logger.error(`Error publishing sync event: ${eventType}`, {
        message: error.message,
        status: error.response?.status,
        data: error.response?.data,
      });
    } else {
      logger.error(`Error publishing sync event: ${eventType}`, { error });
    }
    throw error;
  }
}

/**
 * Publish a config event to Python connector service.
 * Replaces direct Kafka publishing for AI config events.
 * 
 * @param connectorBackend - Python connector service base URL
 * @param eventType - Event type (e.g., 'llmConfigured', 'embeddingModelConfigured')
 * @param payload - Event payload
 */
export async function publishConfigEvent(
  connectorBackend: string,
  eventType: string,
  payload: Record<string, any>
): Promise<void> {
  try {
    const url = `${connectorBackend}/api/v1/connectors/config-event`;
    const data: EventPayload = { eventType, payload };

    logger.info(`Publishing config event to Python: ${eventType}`, {
      url,
      eventType,
    });

    const response = await axios.post(url, data, {
      headers: { 'Content-Type': 'application/json' },
      timeout: 10000,
    });

    if (response.status === 200 && response.data.success) {
      logger.info(`✅ Config event published successfully: ${eventType}`);
    } else {
      logger.error(`Failed to publish config event: ${eventType}`, {
        status: response.status,
        data: response.data,
      });
      throw new Error(`Failed to publish config event: ${eventType}`);
    }
  } catch (error) {
    if (error instanceof AxiosError) {
      logger.error(`Error publishing config event: ${eventType}`, {
        message: error.message,
        status: error.response?.status,
        data: error.response?.data,
      });
    } else {
      logger.error(`Error publishing config event: ${eventType}`, { error });
    }
    throw error;
  }
}

