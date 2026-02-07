import { Container } from 'inversify';
import { Logger } from '../../../libs/services/logger.service';
import { AppConfig } from '../../tokens_manager/config/config';
import { AuthMiddleware } from '../../../libs/middlewares/auth.middleware';
import { AuthTokenService } from '../../../libs/services/authtoken.service';
import { ConfigurationManagerConfig } from '../../configuration_manager/config/config';
import { KeyValueStoreService } from '../../../libs/services/keyValueStore.service';

const loggerConfig = {
  service: 'Crawling Manager Container',
};

/**
 * Crawling Manager Container
 * 
 * NOTE: All crawling scheduling functionality has been moved to the Python connector service
 * which uses APScheduler + Redis. This container now only provides basic authentication
 * middleware for proxying requests to Python.
 */
export class CrawlingManagerContainer {
  private static instance: Container;
  private static logger: Logger = Logger.getInstance(loggerConfig);

  static async initialize(
    configurationManagerConfig: ConfigurationManagerConfig,
    appConfig: AppConfig,
  ): Promise<Container> {
    const container = new Container();
    container.bind<Logger>('Logger').toConstantValue(this.logger);
    container
      .bind<ConfigurationManagerConfig>('ConfigurationManagerConfig')
      .toConstantValue(configurationManagerConfig);

    container
      .bind<AppConfig>('AppConfig')
      .toDynamicValue(() => appConfig)
      .inTransientScope();
    
    await this.initializeServices(container, appConfig, configurationManagerConfig);
    this.instance = container;
    return container;
  }

  private static async initializeServices(
    container: Container,
    appConfig: AppConfig,
    configurationManagerConfig: ConfigurationManagerConfig,
  ): Promise<void> {
    try {
      const logger = container.get<Logger>('Logger');
      logger.info('Initializing Crawling Manager proxy services');

      // Initialize KeyValueStoreService (needed for PrometheusService)
      const keyValueStoreService = KeyValueStoreService.getInstance(
        configurationManagerConfig,
      );
      await keyValueStoreService.connect();
      container
        .bind<KeyValueStoreService>('KeyValueStoreService')
        .toConstantValue(keyValueStoreService);

      // Only initialize authentication middleware for proxying
      const authTokenService = new AuthTokenService(
        appConfig.jwtSecret,
        appConfig.scopedJwtSecret,
      );
      const authMiddleware = new AuthMiddleware(
        container.get('Logger'),
        authTokenService,
      );
      container
        .bind<AuthMiddleware>(AuthMiddleware)
        .toConstantValue(authMiddleware);

      this.logger.info('Crawling Manager proxy services initialized successfully');
    } catch (error) {
      const logger = container.get<Logger>('Logger');
      logger.error('Failed to initialize crawling manager proxy services', {
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  static getInstance(): Container {
    if (!this.instance) {
      throw new Error('Crawling Manager container not initialized');
    }
    return this.instance;
  }

  static async dispose(): Promise<void> {
    if (this.instance) {
      try {
        // Disconnect KeyValueStoreService if bound
        const keyValueStoreService = this.instance.isBound('KeyValueStoreService')
          ? this.instance.get<KeyValueStoreService>('KeyValueStoreService')
          : null;

        if (keyValueStoreService && keyValueStoreService.isConnected()) {
          await keyValueStoreService.disconnect();
          this.logger.info('KeyValueStoreService disconnected successfully');
        }

        this.logger.info('Crawling Manager proxy container disposed');
      } catch (error) {
        this.logger.error(
          'Error while disposing crawling manager container',
          {
            error: error instanceof Error ? error.message : 'Unknown error',
          },
        );
      } finally {
        this.instance = null!;
      }
    }
  }
}
